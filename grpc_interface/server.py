import pickle
from time import sleep
import grpc
from concurrent import futures
import sys
import os
from grpc import StatusCode
import time
from enum import Enum

# Fix for finding the grpc interface
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# gRPC interface
import interface_pb2
import interface_pb2_grpc

# Causal nest
from causal_nest.dataset import (
    Dataset,
    FeatureType,
    FeatureTypeMap,
    MissingDataHandlingMethod,
    handle_missing_data,
    estimate_feature_importances,
)
from causal_nest.problem import Problem
from causal_nest.discovery import (
    DiscoveryResult,
    discover_with_all_models,
    discover_with_model,
    applyable_models,
)

from causal_nest.estimation import EstimationResult, estimate_all_effects
from causal_nest.refutation import refute_all_results
from causal_nest.result import generate_all_results

VERBOSE = int(os.getenv("VERBOSE", 0))

# Creating status ENUM to check connection status
class Status(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Total number of cores avaliable for processing
MAX_CORES = int(os.getenv("MAX_CORES", "6"))
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "30"))

background_executor = futures.ThreadPoolExecutor(max_workers=MAX_CORES)

def print_verbose(*args, **kwargs):
    """
    Verbose print: for debbuging purposes, prints useful information about whats going on.
    """
    if VERBOSE:
        print(*args, **kwargs)


class SerializerServiceServicer(interface_pb2_grpc.SerializerServiceServicer):

    """
    'handle_missing_data_grpc' and 'applyable_models_grpc' are some of the disponible functions that were not implemented
    (as its not being used on the causal nest backend). 

    To implement these functions, just look at: 
    - the interface.proto file for the request and response types
    - the documentation of the causal nest library for the corresponding function.
    - the structure of the implemented functions below (so they stay consistent with all that was already done).
    """
    def handle_missing_data_grpc(self, request, context):
        pass

    def applyable_models_grpc(self, request, context):
        pass


    # IMPLEMENTED FUNCTIONS:

    def testing_connection_grpc(self, request, context):
        print("==========================================")
        print("testing_connection_grpc")
        sleep(5)
        print("complete")
        print("==========================================")
        return interface_pb2.ProblemResponse(
            problem=pickle.dumps("Connection successful!")
        )

    def create_problem_grpc(self, request, context):
        print("==========================================")
        print("create_problem_grpc")

        # Streaming initial response
        yield interface_pb2.CreateProblemResponse(
            problem=pickle.dumps(None), 
            models=pickle.dumps(None),
            status=Status.RUNNING.value
        )
        
        try:
            knowledge = pickle.loads(request.knowledge)
            dataset = pickle.loads(request.dataset)
            feature_mapping = pickle.loads(request.feature_mapping)
            target = request.target
            description = request.description

            # Display information
            print_verbose(" - Knowledge:", knowledge)
            print_verbose(" - Dataset:", dataset)
            print_verbose(" - Feature Mapping:", feature_mapping)
            print_verbose(" - Target:", target)
            print_verbose(" - Description:", description)

            dataset = Dataset(data=dataset, target=target, feature_mapping=feature_mapping)
            dataset = handle_missing_data(dataset, MissingDataHandlingMethod.FORWARD_FILL)
            dataset = estimate_feature_importances(dataset)

            problem: Problem # = None

            if knowledge is None:
                problem = Problem(dataset=dataset, description=description)
            else:
                problem = Problem(
                    dataset=dataset, knowledge=knowledge, description=description
                )

            models = applyable_models(problem)
            if models:
                models = [model.__name__ for model in models]

            print("completed")
            print("==========================================")

            yield interface_pb2.CreateProblemResponse(
                problem=pickle.dumps(problem), 
                models=pickle.dumps(models),
                status=Status.COMPLETED.value
            )

        except Exception as e:
            print("Error during problem creation:", str(e))
            yield interface_pb2.CreateProblemResponse(
                problem=pickle.dumps(None), 
                models=pickle.dumps(None),
                status=Status.FAILED.value
            )

    def discover_with_all_models_grpc(self, request, context):
        print("==========================================")
        print("discover_with_all_models_grpc")

        # Streaming initial response
        yield interface_pb2.ProblemResponse(
            problem=pickle.dumps(None),
            status=Status.RUNNING.value
        )

        try:
            problem = pickle.loads(request.problem)

            # Fixing type to match causal nest function signature
            max_workers = request.max_workers
            if max_workers == 0:
                max_workers = None

            # Display information
            print_verbose(" - Problem:", problem)
            print_verbose(" - Max Seconds Model:", request.max_seconds_model)
            print_verbose(" - Verbose:", request.verbose)
            print_verbose(" - Max Workers:", max_workers)
            print_verbose(" - Orient Toward Target:", request.orient_toward_target)


            future = background_executor.submit(
                discover_with_all_models,
                problem,
                max_seconds_model=request.max_seconds_model,
                verbose=request.verbose,
                # max_workers=max_workers,
                max_workers=MAX_CORES - 1,
                orient_toward_target=request.orient_toward_target,
            )

            while not future.done():
                if not context.is_active():
                    future.cancel()
                    print("Client disconnected, cancelling refutation task.")
                    return 

                yield interface_pb2.ProblemResponse(
                    problem=pickle.dumps(None), 
                    status=Status.RUNNING.value # Keeps sending RUNNING status
                )
                sleep(PING_INTERVAL)

            # Executing finished
            updated_problem:Problem = future.result()

            if updated_problem.discovery_results is None:
                raise Exception("Refutation did not return a result")

            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(updated_problem), 
                status=Status.COMPLETED.value
            )
                
        except Exception as e:
            print("Error during refutation:", str(e))
            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(None),
                status=Status.FAILED.value
            )
            return 

    def estimate_all_effects_grpc(self, request, context):
        print("==========================================")
        print("estimate_all_effects_grpc")
        
        # Streaming initial response
        yield interface_pb2.ProblemResponse(
            problem=pickle.dumps(None),
            status=Status.RUNNING.value
        )
        try:
            problem = pickle.loads(request.problem)
            # Fixing type to match causal nest function signature
            max_workers = request.max_workers
            if max_workers == 0:
                max_workers = None

            # Display information
            print_verbose(" - Problem:", problem)
            print_verbose(" - Max Seconds Model:", request.max_seconds_model)
            print_verbose(" - Verbose:", request.verbose)
            print_verbose(" - Max Workers:", max_workers)

            # updated_problem = estimate_all_effects(
            #     problem,
            #     max_seconds_model=request.max_seconds_model,
            #     verbose=request.verbose,
            #     max_workers=max_workers,
            # )
            future = background_executor.submit(
                estimate_all_effects,
                problem,
                max_seconds_model=request.max_seconds_model,
                verbose=request.verbose,
                max_workers=max_workers,
            )

            while not future.done():
                if not context.is_active():
                    future.cancel()
                    print("Client disconnected, cancelling refutation task.")
                    return 

                yield interface_pb2.ProblemResponse(
                    problem=pickle.dumps(None), 
                    status=Status.RUNNING.value # Keeps sending RUNNING status
                )
                sleep(PING_INTERVAL)

            # Executing finished
            updated_problem:Problem = future.result()

            print("------------------------------------------")
            print("---------- Estimation Results ------------")
            print("------------------------------------------")
            print(updated_problem.estimation_results)
            print("------------------------------------------")
            print("completed")
            print("==========================================")

            if updated_problem.estimation_results is None:
                raise Exception("Refutation did not return a result")

            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(updated_problem), 
                status=Status.COMPLETED.value
            )
                
        except Exception as e:
            print("Error during refutation:", str(e))
            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(None),
                status=Status.FAILED.value
            )
            return 

    def refute_all_results_grpc(self, request, context):
        print("==========================================")
        print("refute_all_results_grpc")

        # Streaming initial response
        yield interface_pb2.ProblemResponse(
            problem=pickle.dumps(None),
            status=Status.RUNNING.value
        )
        try:
            problem = pickle.loads(request.problem)

            # Fixing type to match causal nest function signature
            max_workers = request.max_workers
            if max_workers == 0:
                max_workers = None

            # Display information
            print_verbose(" - Problem:", problem)
            print_verbose(" - Max Seconds Global:", request.max_seconds_global)
            print_verbose(" - Max Seconds Model:", request.max_seconds_model)
            print_verbose(" - Verbose:", request.verbose)
            print_verbose(" - Max Workers:", max_workers)

            future = background_executor.submit(
                refute_all_results,
                problem,
                max_seconds_global=request.max_seconds_global,
                max_seconds_model=request.max_seconds_model,
                verbose=request.verbose,
                max_workers=max_workers,
            )

            while not future.done():
                if not context.is_active():
                    future.cancel()
                    print("Client disconnected, cancelling refutation task.")
                    return 

                yield interface_pb2.ProblemResponse(
                    problem=pickle.dumps(None), 
                    status=Status.RUNNING.value # Keeps sending RUNNING status
                )
                sleep(PING_INTERVAL)

            # Executing finished
            updated_problem:Problem = future.result()
            print("------------------------------------------")
            print("---------- Refutation Results ------------")
            print("------------------------------------------")
            print(updated_problem.refutation_results)
            print("------------------------------------------")
            print("completed")
            print("==========================================")
            while not future.done():
                if not context.is_active():
                    future.cancel()
                    print("Client disconnected, cancelling refutation task.")
                    return 

                yield interface_pb2.ProblemResponse(
                    problem=pickle.dumps(None), 
                    status=Status.RUNNING.value # Keeps sending RUNNING status
                )
                sleep(PING_INTERVAL)

            # Executing finished
            updated_problem = future.result()

            if updated_problem.discovery_results is None:
                raise Exception("Refutation did not return a result")

            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(updated_problem), 
                status=Status.COMPLETED.value
            )
        except Exception as e:
            print("Error during refutation:", str(e))
            yield interface_pb2.ProblemResponse(
                problem=pickle.dumps(None),
                status=Status.FAILED.value
            )
            return 

    def generate_all_results_grpc(self, request, context):
        # Generating the graphs is a rather simple and quick process, so no need for streaming responses here.
        print("==========================================")
        print("generate_all_results_grpc")
        problem = pickle.loads(request.problem)
        if problem.refutation_results is None:
            return context.abort(
                self,
                StatusCode.INTERNAL,
                "Refutation results are required to generate graphs",
            )

        # Fixing type to match causal nest function signature
        layout = request.layout_option
        if layout == "":
            layout = None

        graphs = generate_all_results(
            problem,
            layout_option=layout,
        )
        print("------------------------------------------")
        print("--------- Resultant Graphs JSON ----------")
        print("------------------------------------------")
        print(graphs)
        print("------------------------------------------")
        print("completed")
        print("==========================================")
        return interface_pb2.GraphStringResponse(graph_string=pickle.dumps(graphs))


# GRPC server bootstrap
def serve():

    # In order to maintain long-running connections without interruptions, we set custom gRPC server options.
    server_options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 7200000),
        ("grpc.keepalive_timeout_ms", 100000),
        ("grpc.http2.min_ping_interval_without_data_ms", 1200000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.max_connection_idle_ms", 604800000),
        ("grpc.max_connection_age_ms", 604800000),
        ("grpc.max_connection_age_grace_ms", 900000),
    ]
    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=10),
        options=server_options,
    )
    interface_pb2_grpc.add_SerializerServiceServicer_to_server(
        SerializerServiceServicer(), server
    )
    server.add_insecure_port("[::]:5555")
    server.start()

    print("       ____  ____   ____   ____                             ")
    print("  __ _|  _ \\|  _ \\ / ___| / ___|  ___ _ ____   _____ _ __  ")
    print(" / _` | |_) | |_) | |     \\___ \\ / _ \\ '__\\ \\ / / _ \\ '__|")
    print("| (_| |  _  |  __/| |___   ___) |  __/ |   \\ V /  __/ |  ")
    print(" \\__, |_| \\_\\_|    \\____| |____/ \\___|_|    \\_/ \\___|_|   ")
    print(" |___/                                                    ")

    print("SerializerService running on port 5555...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
# def refute_all_results_grpc(self, request, context):
#     print("==========================================")
#     print("refute_all_results_grpc")
#
#     # Streaming initial response
#     yield interface_pb2.ProblemResponse(
#         problem=b'', 
#         status=Status.RUNNING.value
#     )
#
#     # 1. Prepare parameters
#     try:
#         problem = pickle.loads(request.problem)
#         max_workers = request.max_workers if request.max_workers != 0 else None
#
#         # 2. Submit the long-running function to the thread pool
#         future = background_executor.submit(
#             refute_all_results,
#             problem,
#             max_seconds_global=request.max_seconds_global,
#             max_seconds_model=request.max_seconds_model,
#             verbose=request.verbose,
#             max_workers=max_workers,
#         )
#
#         # 3. Intermediate Yielding Loop (The "Ping" Mechanism)
#         # We will yield a status update every 30 seconds to maintain the connection.
#         ping_interval_seconds = 30
#
#         while not future.done():
#             # Check if the RPC is still active (client hasn't cancelled)
#             if not context.is_active():
#                 # Client disconnected, cancel the future and break
#                 future.cancel()
#                 print("Client disconnected, cancelling refutation task.")
#                 return 
#
#             # Yield an intermediate status update to reset all network/gRPC idle timers
#             yield interface_pb2.ProblemResponse(
#                 problem=b'', 
#                 status=Status.RUNNING.value # Keep sending RUNNING status
#             )
#
#             sleep(ping_interval_seconds) # Wait before checking the task and pinging again
#
#         # 4. Process the Result
#         updated_problem = future.result() # Get the result (this will raise any exceptions from the background thread)
#
#         # ... print statements ...
#
#         if updated_problem.refutation_results is None:
#             raise Exception("Refutation did not return a result")
#
#         # Final success yield
#         yield interface_pb2.ProblemResponse(
#             problem=pickle.dumps(updated_problem), 
#             status=Status.COMPLETED.value
#         )
#
#     except Exception as e:
#         print("Error during refutation:", str(e))
#
#         # Yield FAILED status
#         yield interface_pb2.ProblemResponse(
#             problem=b'',
#             status=Status.FAILED.value
#         )
#         return # Terminate stream after failure yield

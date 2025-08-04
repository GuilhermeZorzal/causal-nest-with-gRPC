import pickle
from time import sleep
import grpc
from concurrent import futures
import sys
import os
from grpc import StatusCode

# gRPC interface
import interface_pb2
import interface_pb2_grpc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Native causal nest
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

from causal_nest.estimation import (
    EstimationResult,
    estimate_all_effects,
)
from causal_nest.refutation import (
    refute_all_results,
)
from causal_nest.result import (
    generate_all_results,
)


class SerializerServiceServicer(interface_pb2_grpc.SerializerServiceServicer):
    def testing_connection_grpc(self, request, context):
        print("testing_connection_grpc")
        sleep(10)
        print("testing_connection_grpc")
        return interface_pb2.ProblemResponse(
            problem=pickle.dumps("estranhamente conectou ao grpc")
        )

    def handle_missing_data_grpc(self, request, context):
        print("handle_missing_data_grpc")
        dataset: Dataset = pickle.loads(request.dataset)
        method: MissingDataHandlingMethod = pickle.loads(
            request.missing_data_handling_method
        )
        if method is None:
            method = MissingDataHandlingMethod.DROP
        result: Dataset = handle_missing_data(dataset=dataset, method=method)
        print("completed")

        return interface_pb2.DatasetResponse(dataset=pickle.dumps(result))

    def applyable_models_grpc(self, request, context):
        print("applyable_models_grpc")
        problem: Problem = pickle.loads(request.problem)
        model_list = applyable_models(problem)
        print("completed")
        return interface_pb2.ModelsResponse(model_names=pickle.dumps(model_list))

    def create_problem_grpc(self, request, context):
        print("create_problem_grpc")
        knowledge = pickle.loads(request.knowledge)
        dataset = pickle.loads(request.dataset)
        feature_mapping = pickle.loads(request.feature_mapping)
        target = request.target
        description = request.description

        dataset = Dataset(data=dataset, target=target, feature_mapping=feature_mapping)
        dataset = handle_missing_data(dataset, MissingDataHandlingMethod.FORWARD_FILL)
        dataset = estimate_feature_importances(dataset)

        problem = None

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

        return interface_pb2.CreateProblemResponse(
            problem=pickle.dumps(problem), models=pickle.dumps(models)
        )

    def discover_with_all_models_grpc(self, request, context):
        print("discover_with_all_models_grpc")
        problem = pickle.loads(request.problem)

        max_workers = request.max_workers
        if max_workers == 0:
            max_workers = None
        updated_problem = discover_with_all_models(
            problem,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=max_workers,
            orient_toward_target=request.orient_toward_target,
        )

        print("------------------------------------------")
        print("----------- Discovery Results ------------")
        print("------------------------------------------")
        print(updated_problem.discovery_results)
        print("------------------------------------------")

        print("completed")
        if updated_problem.discovery_results is None:
            return context.abort(
                self, StatusCode.INTERNAL, "Discovery did not return a result"
            )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def estimate_all_effects_grpc(self, request, context):
        print("estimate_all_effects_grpc")
        problem = pickle.loads(request.problem)

        max_workers = request.max_workers
        if max_workers == 0:
            max_workers = None
        updated_problem = estimate_all_effects(
            problem,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=max_workers,
        )
        print("------------------------------------------")
        print("---------- Estimation Results ------------")
        print("------------------------------------------")
        print(updated_problem.estimation_results)
        print("------------------------------------------")
        print("completed")
        if updated_problem.estimation_results is None:
            return context.abort(
                self, StatusCode.INTERNAL, "Estimation did not return a result"
            )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def refute_all_results_grpc(self, request, context):
        print("refute_all_results_grpc")
        problem = pickle.loads(request.problem)
        max_workers = request.max_workers
        if max_workers == 0:
            max_workers = None
        updated_problem = refute_all_results(
            problem,
            max_seconds_global=request.max_seconds_global,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=max_workers,
        )
        print("------------------------------------------")
        print("---------- Refutation Results ------------")
        print("------------------------------------------")
        print(updated_problem.refutation_results)
        print("------------------------------------------")
        print("completed")
        if updated_problem.refutation_results is None:
            return context.abort(
                self, StatusCode.INTERNAL, "Refutation did not return a result"
            )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def generate_all_results_grpc(self, request, context):
        print("generate_all_results_grpc")
        problem = pickle.loads(request.problem)
        if problem.refutation_results is None:
            return context.abort(
                self,
                StatusCode.INTERNAL,
                "Refutation results are required to generate graphs",
            )
        layout = request.layout_option
        if layout == "":
            layout = None
        graphs = generate_all_results(
            problem,
            layout_option=layout,
        )
        print("------------------------------------------")
        print("---------- Resultant Graphs ------------")
        print("------------------------------------------")
        print(graphs)
        print("------------------------------------------")
        print("completed")
        return interface_pb2.GraphStringResponse(graph_string=pickle.dumps(graphs))


# GRPC server bootstrap
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    interface_pb2_grpc.add_SerializerServiceServicer_to_server(
        SerializerServiceServicer(), server
    )
    server.add_insecure_port("[::]:5555")
    server.start()
    print("SerializerService running on port 5555...")
    server.wait_for_termination()


print("Starting gRPC server...")

if __name__ == "__main__":
    serve()

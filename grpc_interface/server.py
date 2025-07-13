import pickle
import grpc
from concurrent import futures
import sys
import os

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
    estimate_effect,
)
from causal_nest.refutation import (
    refute_all_results,
    refute_with_model,
)
from causal_nest.result import (
    generate_all_results,
)

# from causal_nest import (
#     handle_missing_data,
#     applyable_models,
#     discover_with_all_models,
#     estimate_all_effects,
#     refute_all_results,
#     generate_all_results,
# )


class SerializerServiceServicer(interface_pb2_grpc.SerializerServiceServicer):
    def handle_missing_data_grpc(self, request, context):
        dataset: Dataset = pickle.loads(request.dataset)
        method: MissingDataHandlingMethod = pickle.loads(
            request.missing_data_handling_method
        )
        if method is None:
            method = MissingDataHandlingMethod.DROP
        result: Dataset = handle_missing_data(dataset=dataset, method=method)

        return interface_pb2.DatasetResponse(dataset=pickle.dumps(result))

    def applyable_models_grpc(self, request, context):
        problem: Problem = pickle.loads(request.problem)
        model_list = applyable_models(problem)
        return interface_pb2.ModelsResponse(model_names=pickle.dumps(model_list))

    def discover_with_all_models_grpc(self, request, context):
        problem = pickle.loads(request.problem)
        updated_problem = discover_with_all_models(
            problem,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=request.max_workers,
            orient_toward_target=request.orient_toward_target,
        )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def estimate_all_effects_grpc(self, request, context):
        problem = pickle.loads(request.problem)
        updated_problem = estimate_all_effects(
            problem,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=request.max_workers,
        )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def refute_all_results_grpc(self, request, context):
        problem = pickle.loads(request.problem)
        updated_problem = refute_all_results(
            problem,
            max_seconds_global=request.max_seconds_global,
            max_seconds_model=request.max_seconds_model,
            verbose=request.verbose,
            max_workers=request.max_workers,
        )
        return interface_pb2.ProblemResponse(problem=pickle.dumps(updated_problem))

    def generate_all_results_grpc(self, request, context):
        problem = pickle.loads(request.problem)
        graph_string = generate_all_results(
            problem,
            layout_option=request.layout_option,
        )
        return interface_pb2.GraphStringResponse(graph_string=graph_string)


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

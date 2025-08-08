# gRPC Interface

This is the file that describes the gRPC interface for the `grpc_interface` module.

## File structure
.
├── interface.proto - interface definition file
├── Makefile - makefile to generate the gRPC interface files
├── interface_pb2_grpc.py - generated gRPC interface file (created through `make`)
├── interface_pb2.py - generated gRPC interface file (created through `make`)
├── README.md
└── server.py - grpc server implementation 

The files you should take a look to understand the gRPC interface are:
- `interface.proto`: This file has the definition of the gRPC interface, including the messages and their fields.
- `server.py`: This file has the implementation of the gRPC server.

Basically, the `server.py` file has the same signatures of the original `causal_nest` library. The server just calls the causal_nest methods and returns the results.


## Next steps

For now, only the essential functions for the causal nest web application are implemented. Here is a list of the functions to be implemented:

```
rpc discover_with_model_grpc(DiscoverWithModelRequest) returns (DiscoveryResultResponse);

// Estimation
rpc estimate_model_effects_grpc(EstimateModelEffectsRequest) returns (EstimationResultResponse);

// Refutation
rpc refute_with_model_grpc(RefuteWithModelRequest) returns (RefutationResultResponse);

// Results
rpc generate_result_graph_grpc(GenerateResultGraphRequest) returns (GraphStringResponse);
```

The definitions of these functions can be found in the `interface.proto` file. The implementation of these functions should be done in the `server.py` file, following the same structure as the existing functions.

namespace KnowledgeOps.Infrastructure;

public sealed record QdrantVectorStoreOptions(string Host, int GrpcPort, string CollectionName, int VectorSize);

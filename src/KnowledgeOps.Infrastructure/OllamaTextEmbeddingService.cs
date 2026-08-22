using System.Net;
using KnowledgeOps.Application;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using OllamaSharp.Models.Exceptions;

namespace KnowledgeOps.Infrastructure;

public sealed class OllamaTextEmbeddingService(
    IEmbeddingGenerator<string, Embedding<float>> generator, IOptions<EmbeddingOptions> options) : ITextEmbeddingService {
    private const string QUERY_PREFIX = "task: search result | query: ";
    private const string DOCUMENT_PREFIX = "title: none | text: ";

    private readonly EmbeddingOptions _options = options.Value;

    public async Task<float[]> EmbedQueryAsync(string query, CancellationToken cancellationToken = default) {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);

        var vectors = await GenerateAsync([QUERY_PREFIX + query], cancellationToken);
        return vectors[0];
    }

    public async Task<IReadOnlyList<float[]>> EmbedDocumentsAsync(
        IReadOnlyList<string> texts, CancellationToken cancellationToken = default) {
        ArgumentNullException.ThrowIfNull(texts);
        if (texts.Count == 0) {
            return [];
        }

        var inputs = new string[texts.Count];
        for (int i = 0; i < inputs.Length; i++) {
            string text = texts[i];
            if (string.IsNullOrWhiteSpace(text)) {
                throw new ArgumentException($"Document text at index {i} must contain non-whitespace characters.", nameof(texts));
            }

            inputs[i] = DOCUMENT_PREFIX + text;
        }

        return await GenerateAsync(inputs, cancellationToken);
    }

    private async Task<float[][]> GenerateAsync(string[] inputs, CancellationToken cancellationToken) {
        cancellationToken.ThrowIfCancellationRequested();
        GeneratedEmbeddings<Embedding<float>> embeddings;
        try {
            embeddings = await generator.GenerateAsync(inputs, cancellationToken: cancellationToken);
        } catch (HttpRequestException exception) when (exception.StatusCode == HttpStatusCode.NotFound) {
            throw new InvalidOperationException(
                $"Ollama at '{_options.BaseUrl}' could not find model '{_options.Model}'. Check the URL and run 'ollama pull {_options.Model}'.",
                exception);
        } catch (HttpRequestException exception) {
            throw new InvalidOperationException(
                $"Could not generate embeddings with Ollama at '{_options.BaseUrl}' using model '{_options.Model}'.",
                exception);
        } catch (OllamaException exception) {
            throw new InvalidOperationException(
                $"Ollama rejected embedding generation for model '{_options.Model}'. {exception.Message}", exception);
        }

        if (embeddings.Count != inputs.Length) {
            throw new InvalidOperationException(
                $"Ollama returned {embeddings.Count} embeddings for {inputs.Length} inputs.");
        }

        var vectors = new float[embeddings.Count][];
        for (int i = 0; i < vectors.Length; i++) {
            var vector = embeddings[i].Vector;
            ValidateVector(vector.Span, i);
            vectors[i] = vector.ToArray();
        }

        return vectors;
    }

    private void ValidateVector(ReadOnlySpan<float> vector, int index) {
        if (vector.Length != _options.VectorSize) {
            throw new InvalidOperationException(
                $"Ollama embedding at index {index} has dimension {vector.Length}, expected {_options.VectorSize}.");
        }

        bool hasNonZeroValue = false;
        foreach (float value in vector) {
            if (!float.IsFinite(value)) {
                throw new InvalidOperationException($"Ollama embedding at index {index} must contain only finite values.");
            }

            hasNonZeroValue |= value != 0f;
        }

        if (!hasNonZeroValue) {
            throw new InvalidOperationException($"Ollama embedding at index {index} must have a non-zero norm.");
        }
    }
}

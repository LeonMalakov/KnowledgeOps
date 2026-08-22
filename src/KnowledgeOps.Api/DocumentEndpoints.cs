using KnowledgeOps.Application;
using Microsoft.AspNetCore.Mvc;

namespace KnowledgeOps.Api;

internal static class DocumentEndpoints {
    private const int MAX_MULTIPART_BYTES = IDocumentTextExtractor.MAX_FILE_BYTES + 64 * 1024;

    public static void MapDocumentEndpoints(this WebApplication app) {
        app.MapPost("/api/documents", IndexDocumentAsync);
        app.MapPost("/api/documents/upload", UploadDocumentAsync)
            .WithMetadata(new RequestSizeLimitAttribute(MAX_MULTIPART_BYTES), new RequestFormLimitsAttribute {
                ValueCountLimit = 3,
                MultipartBodyLengthLimit = IDocumentTextExtractor.MAX_FILE_BYTES,
                MemoryBufferThreshold = IDocumentTextExtractor.MAX_FILE_BYTES
            })
            // This API is anonymous and uses no auth cookies. The upload handler rejects foreign browser origins.
            .DisableAntiforgery();
        app.MapPut("/api/documents/{documentId:guid}", UpdateDocumentAsync);
        app.MapDelete("/api/documents/{documentId:guid}", DeleteDocumentAsync);
        app.MapPost("/api/search", SearchAsync);
        app.MapPost("/api/search/hybrid", SearchHybridAsync);
    }

    private static async Task<IResult> UploadDocumentAsync(
        HttpRequest request, IFormCollection form, IDocumentTextExtractor extractor,
        DocumentIndexingService indexing, CancellationToken cancellationToken) {
        if (IsCrossOrigin(request)) {
            return TypedResults.Problem(title: "Cross-origin file uploads are not allowed.", statusCode: StatusCodes.Status403Forbidden);
        }

        if (!string.Equals(request.GetTypedHeaders().ContentType?.MediaType.Value, "multipart/form-data", StringComparison.OrdinalIgnoreCase)) {
            return TypedResults.Problem(title: "Use multipart/form-data to upload a file.", statusCode: StatusCodes.Status415UnsupportedMediaType);
        }

        var file = form.Files.GetFile("file");
        if (file is null || form.Files.Count != 1) {
            return FileProblem("Supply exactly one file in the 'file' field.");
        }

        if (file.Length == 0) {
            return FileProblem("The file is empty.");
        }

        if (file.Length > IDocumentTextExtractor.MAX_FILE_BYTES) {
            return TypedResults.Problem(title: "The file exceeds the 5 MiB limit.", statusCode: StatusCodes.Status413PayloadTooLarge);
        }

        string fileName = Path.GetFileName(file.FileName.Replace('\\', '/'));
        string title = form.TryGetValue("title", out var suppliedTitle) ? suppliedTitle.ToString() : Path.GetFileNameWithoutExtension(fileName);
        string? source = form.TryGetValue("source", out var suppliedSource) ? suppliedSource.ToString() : null;
        if (ValidateDocumentMetadata(title, source) is { } metadataError) {
            return metadataError;
        }

        string content;
        await using (var stream = file.OpenReadStream()) {
            try {
                content = await extractor.ExtractAsync(fileName, stream, cancellationToken);
            } catch (NotSupportedException exception) {
                return TypedResults.Problem(title: exception.Message, statusCode: StatusCodes.Status415UnsupportedMediaType);
            } catch (DocumentTextExtractionException exception) {
                return FileProblem(exception.Message);
            }
        }

        if (ValidateDocumentContent(content) is { } contentError) {
            return contentError;
        }

        return TypedResults.Ok(await indexing.IndexAsync(title, content, source, cancellationToken));
    }

    private static bool IsCrossOrigin(HttpRequest request) {
        if (string.Equals(request.Headers["Sec-Fetch-Site"].ToString(), "cross-site", StringComparison.OrdinalIgnoreCase)) {
            return true;
        }

        if (!request.Headers.TryGetValue("Origin", out var origin)) {
            return false;
        }

        return !Uri.TryCreate(origin.ToString(), UriKind.Absolute, out var originUri)
            || !Uri.TryCreate($"{request.Scheme}://{request.Host.ToUriComponent()}", UriKind.Absolute, out var requestUri)
            || !string.Equals(originUri.GetLeftPart(UriPartial.Authority), requestUri.GetLeftPart(UriPartial.Authority), StringComparison.OrdinalIgnoreCase);
    }

    private static IResult FileProblem(string message) {
        return TypedResults.ValidationProblem(new Dictionary<string, string[]> { ["file"] = [message] });
    }

    private static async Task<IResult> IndexDocumentAsync(
        DocumentRequest request, DocumentIndexingService indexing, CancellationToken cancellationToken) {
        if (ValidateDocument(request) is { } error) {
            return error;
        }

        return TypedResults.Ok(await indexing.IndexAsync(request.Title!, request.Content!, request.Source, cancellationToken));
    }

    private static async Task<IResult> UpdateDocumentAsync(
        Guid documentId, DocumentRequest request, DocumentIndexingService indexing,
        ILogger<DocumentIndexingService> logger, CancellationToken cancellationToken) {
        if (documentId == Guid.Empty) {
            return InvalidDocumentId();
        }

        if (ValidateDocument(request) is { } error) {
            return error;
        }

        try {
            var result = await indexing.UpdateAsync(documentId, request.Title!, request.Content!, request.Source, cancellationToken);
            return result is null ? TypedResults.NotFound() : TypedResults.Ok(result);
        } catch (DocumentUpdateCleanupException exception) {
            logger.LogError(exception, "Update of document {DocumentId} is in an uncertain state after writing its new chunks.", documentId);
            cancellationToken.ThrowIfCancellationRequested();
            return TypedResults.Problem(
                title: "Document update could not be confirmed.",
                detail: exception.Message,
                statusCode: StatusCodes.Status500InternalServerError);
        }
    }

    private static IResult? ValidateDocument(DocumentRequest request) {
        return ValidateDocumentMetadata(request.Title, request.Source) ?? ValidateDocumentContent(request.Content);
    }

    private static IResult? ValidateDocumentMetadata(string? title, string? source) {
        if (string.IsNullOrWhiteSpace(title) || title.Length > DocumentIndexingService.MAX_TITLE_LENGTH) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["title"] = [$"Title must contain non-whitespace characters and be at most {DocumentIndexingService.MAX_TITLE_LENGTH} characters."]
            });
        }

        if (source?.Length > DocumentIndexingService.MAX_SOURCE_LENGTH) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["source"] = [$"Source must not exceed {DocumentIndexingService.MAX_SOURCE_LENGTH} characters."]
            });
        }

        return null;
    }

    private static IResult? ValidateDocumentContent(string? content) {
        if (string.IsNullOrWhiteSpace(content)) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["content"] = ["Content must contain non-whitespace characters."]
            });
        }

        if (content.Length > DocumentIndexingService.MAX_CONTENT_LENGTH) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["content"] = [$"Content must not exceed {DocumentIndexingService.MAX_CONTENT_LENGTH} characters."]
            });
        }

        return null;
    }

    private static async Task<IResult> DeleteDocumentAsync(
        Guid documentId, DocumentIndexingService indexing, CancellationToken cancellationToken) {
        if (documentId == Guid.Empty) {
            return InvalidDocumentId();
        }

        await indexing.DeleteDocumentAsync(documentId, cancellationToken);
        return TypedResults.NoContent();
    }

    private static IResult InvalidDocumentId() {
        return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
            ["documentId"] = ["Document ID must be non-empty."]
        });
    }

    private static async Task<IResult> SearchAsync(
        SearchRequest request, SemanticSearchService search, CancellationToken cancellationToken) {
        if (ValidateSearch(request) is { } error) {
            return error;
        }

        return TypedResults.Ok(await search.SearchAsync(request.Query!, request.TopK, cancellationToken));
    }

    private static async Task<IResult> SearchHybridAsync(
        SearchRequest request, HybridSearchService search, CancellationToken cancellationToken) {
        if (ValidateSearch(request) is { } error) {
            return error;
        }

        return TypedResults.Ok(await search.SearchAsync(request.Query!, request.TopK, cancellationToken));
    }

    private static IResult? ValidateSearch(SearchRequest request) {
        if (string.IsNullOrWhiteSpace(request.Query)) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["query"] = ["Query must contain non-whitespace characters."]
            });
        }

        if (request.TopK is < SemanticSearchService.MIN_TOP_K or > SemanticSearchService.MAX_TOP_K) {
            return TypedResults.ValidationProblem(new Dictionary<string, string[]> {
                ["topK"] = [$"TopK must be between {SemanticSearchService.MIN_TOP_K} and {SemanticSearchService.MAX_TOP_K}."]
            });
        }

        return null;
    }

    private sealed record DocumentRequest(string? Title, string? Content, string? Source);

    private sealed record SearchRequest(string? Query, int TopK);
}

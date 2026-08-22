namespace KnowledgeOps.Evaluation;

internal sealed record EvaluationDocument(string Key, string Title, string Content);

internal sealed record EvaluationQuestion(string Query, string[] RelevantDocuments, bool Answerable);

internal sealed record EvaluationDataset(EvaluationDocument[] Documents, EvaluationQuestion[] Questions) {
    public void Validate() {
        if (Documents is not { Length: > 0 } || Questions is not { Length: > 0 }) {
            throw new InvalidDataException("The dataset must contain documents and questions.");
        }

        var keys = new HashSet<string>(StringComparer.Ordinal);
        foreach (var document in Documents) {
            if (string.IsNullOrWhiteSpace(document.Key) || string.IsNullOrWhiteSpace(document.Title)
                || string.IsNullOrWhiteSpace(document.Content)
                || !keys.Add(document.Key)) {
                throw new InvalidDataException("Each document must have a unique non-empty key, a title, and content.");
            }
        }

        foreach (var question in Questions) {
            if (string.IsNullOrWhiteSpace(question.Query) || question.RelevantDocuments is null
                || question.Answerable != (question.RelevantDocuments.Length > 0)
                || question.RelevantDocuments.Any(key => !keys.Contains(key))) {
                throw new InvalidDataException($"Invalid query or relevance labels for '{question.Query}'.");
            }
        }
    }
}

internal sealed record RankedDocument(string Key, float Score);

internal sealed record QuestionResult(
    string Query, bool Answerable, string[] RelevantDocuments, RankedDocument[] FoundDocuments,
    int? FirstRelevantRank, float? TopScore);

internal sealed record EvaluationReport(
    string Dataset, string Mode, string Model, int VectorSize, int RetrievalChunkLimit, int DocumentCount, int QuestionCount,
    int AnswerableCount, int UnanswerableCount, RetrievalMetrics Metrics, QuestionResult[] Questions);

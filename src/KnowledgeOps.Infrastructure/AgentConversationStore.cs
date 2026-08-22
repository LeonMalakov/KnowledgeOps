using Microsoft.Extensions.AI;

namespace KnowledgeOps.Infrastructure;

public sealed record AgentPendingAction(Guid ActionId, string OrderId, string Issue, DateTimeOffset ExpiresAt);

public sealed record AgentActionResult(Guid ConversationId, Guid ActionId, string Status, Guid? TicketId = null);

public sealed class AgentConversationStore(TimeProvider clock) {
    private const int MAX_CONVERSATIONS = 1_024;
    private static readonly TimeSpan _idleLifetime = TimeSpan.FromMinutes(30);
    private readonly Lock _gate = new();
    private readonly Dictionary<Guid, Conversation> _conversations = [];

    internal DateTimeOffset Now => clock.GetUtcNow();

    internal Lease Open(Guid? conversationId, CancellationToken cancellationToken) {
        lock (_gate) {
            cancellationToken.ThrowIfCancellationRequested();
            var now = Now;
            foreach (var (id, entry) in _conversations) {
                if (!entry.Busy && now - entry.LastActivity >= _idleLifetime) {
                    _conversations.Remove(id);
                }
            }

            if (conversationId is null) {
                if (_conversations.Count >= MAX_CONVERSATIONS) {
                    throw new AgentException(AgentFailure.Capacity, "The in-memory conversation limit has been reached.");
                }

                var created = new Conversation(Guid.NewGuid());
                _conversations.Add(created.Id, created);
                return new Lease(this, created);
            }

            if (!_conversations.TryGetValue(conversationId.Value, out var conversation)) {
                throw new AgentException(AgentFailure.NotFound, "Conversation not found or expired.");
            }

            if (conversation.Busy) {
                throw new AgentException(AgentFailure.Conflict, "Another request is using this conversation.");
            }

            return new Lease(this, conversation);
        }
    }

    public AgentActionResult Decide(Guid conversationId, Guid actionId, bool confirmed, CancellationToken cancellationToken) {
        using var lease = Open(conversationId, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return lease.Conversation.Decide(actionId, confirmed, Now);
    }

    internal sealed class Lease : IDisposable {
        private readonly AgentConversationStore _store;
        public Conversation Conversation { get; }

        public Lease(AgentConversationStore store, Conversation conversation) {
            _store = store;
            Conversation = conversation;
            conversation.Busy = true;
        }

        public void Dispose() {
            lock (_store._gate) {
                Conversation.LastActivity = _store.Now;
                Conversation.Busy = false;
            }
        }
    }

    internal sealed class Conversation(Guid id) {
        private const int MAX_HISTORY_TURNS = 6;
        private const int MAX_HISTORY_CHARACTERS = 12_000;
        private const int MAX_ACTIONS = 64;
        private static readonly TimeSpan _actionLifetime = TimeSpan.FromMinutes(5);
        private readonly Queue<(ChatMessage[] Messages, int Size)> _turns = new();
        private readonly Dictionary<Guid, ActionEntry> _actions = [];
        private int _historySize;
        private ActionEntry? _latestAction;

        public Guid Id { get; } = id;
        public bool Busy { get; set; }
        public DateTimeOffset LastActivity { get; set; }
        public IEnumerable<ChatMessage> History => _turns.SelectMany(turn => turn.Messages);

        public AgentPendingAction? GetPending(DateTimeOffset now) =>
            _latestAction is { Decision: null } entry && entry.Action.ExpiresAt > now ? entry.Action : null;

        public string DescribeLatestAction(DateTimeOffset now) {
            return _latestAction switch {
                { Ticket: { } ticket } => $"Последняя заявка для {ticket.OrderId} подтверждена. Номер заявки: {ticket.Id}.",
                { Decision: false } entry => $"Пользователь отклонил заявку для {entry.Action.OrderId}.",
                { } entry when entry.Action.ExpiresAt <= now => $"Срок подтверждения заявки для {entry.Action.OrderId} истёк.",
                { } entry => $"Заявка для {entry.Action.OrderId} ожидает подтверждения через API.",
                _ => ""
            };
        }

        public AgentPendingAction? Prepare(string orderId, string issue, DateTimeOffset now) =>
            _actions.Count >= MAX_ACTIONS ? null : new AgentPendingAction(Guid.NewGuid(), orderId, issue, now + _actionLifetime);

        public void SaveTurn(ChatMessage[] messages, int size, AgentPendingAction? prepared) {
            if (prepared is not null) {
                _latestAction = new ActionEntry(prepared);
                _actions.Add(prepared.ActionId, _latestAction);
            }

            _turns.Enqueue((messages, size));
            _historySize += size;
            while (_turns.Count > MAX_HISTORY_TURNS || _historySize > MAX_HISTORY_CHARACTERS) {
                _historySize -= _turns.Dequeue().Size;
            }
        }

        public AgentActionResult Decide(Guid actionId, bool confirmed, DateTimeOffset now) {
            if (!_actions.TryGetValue(actionId, out var entry)) {
                throw new AgentException(AgentFailure.NotFound, "Pending action not found in this conversation.");
            }

            if (entry.Decision is bool decision) {
                if (decision != confirmed) {
                    throw new AgentException(AgentFailure.Conflict, "The action already has a different decision.");
                }

                return Result(entry);
            }

            if (entry.Action.ExpiresAt <= now) {
                throw new AgentException(AgentFailure.Expired, "The confirmation deadline has passed.");
            }

            if (confirmed) {
                entry.Ticket = new SupportTicket(Guid.NewGuid(), entry.Action.OrderId, entry.Action.Issue, now);
            }

            entry.Decision = confirmed;
            return Result(entry);
        }

        private AgentActionResult Result(ActionEntry entry) =>
            new(Id, entry.Action.ActionId, entry.Ticket is null ? "Rejected" : "Confirmed", entry.Ticket?.Id);
    }

    private sealed class ActionEntry(AgentPendingAction action) {
        public AgentPendingAction Action { get; } = action;
        public bool? Decision { get; set; }
        public SupportTicket? Ticket { get; set; }
    }

    private sealed record SupportTicket(Guid Id, string OrderId, string Issue, DateTimeOffset CreatedAt);
}

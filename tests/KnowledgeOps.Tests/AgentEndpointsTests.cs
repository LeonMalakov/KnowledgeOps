using System.Net;
using System.Net.Http.Json;
using KnowledgeOps.Application;
using KnowledgeOps.Infrastructure;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace KnowledgeOps.Tests;

public class AgentEndpointsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>> {
    [Fact]
    public async Task PostRun_ReturnsAnswerAndActualToolCalls() {
        using var chat = new ScriptedAgentChatClient(call => call == 1
            ? AgentServiceTests.ToolResponse(new FunctionCallContent("order", "get_order_status",
                new Dictionary<string, object?> { ["orderId"] = "ORD-123" }))
            : AgentServiceTests.Answer("Заказ в обработке."));
        using var app = WithFakes(chat);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/agent/run", new { message = "Где заказ ORD-123?" });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<AgentResult>();
        Assert.NotNull(result);
        Assert.Equal("Заказ в обработке.", result.Answer);
        Assert.Equal(new AgentToolCall("get_order_status", true), Assert.Single(result.ToolCalls));
        Assert.NotEqual(Guid.Empty, result.ConversationId);
        Assert.Null(result.PendingAction);
    }

    [Fact]
    public async Task PostRunAndConfirm_RequiresExplicitApprovalAndReturnsSameTicketOnRetry() {
        using var chat = new ScriptedAgentChatClient(call => call == 2
            ? AgentConversationTests.Ticket("action", "ORD-123")
            : AgentServiceTests.Answer("Ответ"));
        using var app = WithFakes(chat);
        using var client = app.CreateClient();
        using var first = await client.PostAsJsonAsync("/api/agent/run", new { conversationId = (Guid?)null, message = "Где ORD-123?" });
        first.EnsureSuccessStatusCode();
        var firstTurn = Assert.IsType<AgentResult>(await first.Content.ReadFromJsonAsync<AgentResult>());
        var conversationId = firstTurn.ConversationId;
        using var second = await client.PostAsJsonAsync("/api/agent/run", new { conversationId, message = "Создай заявку на его отмену." });
        second.EnsureSuccessStatusCode();
        var secondTurn = Assert.IsType<AgentResult>(await second.Content.ReadFromJsonAsync<AgentResult>());
        Assert.Equal(conversationId, secondTurn.ConversationId);
        var actionId = Assert.IsType<AgentPendingAction>(secondTurn.PendingAction).ActionId;

        using var unspecified = await client.PostAsJsonAsync("/api/agent/actions/confirm", new { conversationId, actionId });
        Assert.Equal(HttpStatusCode.BadRequest, unspecified.StatusCode);
        using var confirmed = await client.PostAsJsonAsync("/api/agent/actions/confirm", new { conversationId, actionId, confirmed = true });
        using var repeat = await client.PostAsJsonAsync("/api/agent/actions/confirm", new { conversationId, actionId, confirmed = true });
        Assert.Equal(HttpStatusCode.OK, confirmed.StatusCode);
        Assert.Equal(HttpStatusCode.OK, repeat.StatusCode);
        var ticket = Assert.IsType<AgentActionResult>(await confirmed.Content.ReadFromJsonAsync<AgentActionResult>());
        Assert.NotNull(ticket.TicketId);
        Assert.Equal(ticket, await repeat.Content.ReadFromJsonAsync<AgentActionResult>());
        Assert.Equal(3, chat.Messages.Count);
    }

    [Fact]
    public async Task PostRun_Returns404ForUnknownConversation() {
        using var chat = new ScriptedAgentChatClient(_ => AgentServiceTests.Answer("Unused"));
        using var app = WithFakes(chat);
        using var client = app.CreateClient();
        using var response = await client.PostAsJsonAsync("/api/agent/run", new { conversationId = Guid.NewGuid(), message = "Вопрос" });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Empty(chat.Messages);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" ")]
    public async Task PostRun_InvalidMessageReturns400(string? message) {
        using var chat = new ScriptedAgentChatClient(_ => AgentServiceTests.Answer("Unused"));
        using var app = WithFakes(chat);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/agent/run", new { message });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Empty(chat.Messages);
    }

    [Theory]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    [InlineData(HttpStatusCode.GatewayTimeout)]
    [InlineData(HttpStatusCode.BadGateway)]
    public async Task PostRun_ModelErrorsReturnSanitizedProblem(HttpStatusCode expected) {
        using var chat = new ScriptedAgentChatClient(_ => throw expected switch {
            HttpStatusCode.ServiceUnavailable => new HttpRequestException("private provider details"),
            HttpStatusCode.GatewayTimeout => new TaskCanceledException("private provider details"),
            _ => new InvalidOperationException("private provider details")
        });
        using var app = WithFakes(chat);
        using var client = app.CreateClient();

        using var response = await client.PostAsJsonAsync("/api/agent/run", new { message = "Question" });

        Assert.Equal(expected, response.StatusCode);
        Assert.Contains("Ollama", (await response.Content.ReadFromJsonAsync<ProblemDetails>())?.Title);
        Assert.DoesNotContain("private provider details", await response.Content.ReadAsStringAsync());
    }

    private WebApplicationFactory<Program> WithFakes(IChatClient chat) =>
        factory.WithWebHostBuilder(builder => builder.ConfigureTestServices(services => {
            services.RemoveAll<IChatClient>();
            services.RemoveAll<ITextEmbeddingService>();
            services.RemoveAll<IVectorStore>();
            services.RemoveAll<ITextReranker>();
            services.AddSingleton(chat);
            services.AddSingleton<ITextEmbeddingService>(new FakeTextEmbeddingService());
            services.AddSingleton<IVectorStore>(new FakeVectorStore());
            services.AddSingleton<ITextReranker>(new FakeTextReranker());
        }));
}

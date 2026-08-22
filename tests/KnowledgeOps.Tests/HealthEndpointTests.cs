using System.Net;
using Microsoft.AspNetCore.Mvc.Testing;

namespace KnowledgeOps.Tests;

public class HealthEndpointTests(WebApplicationFactory<Program> factory)
    : IClassFixture<WebApplicationFactory<Program>> {
    [Fact]
    public async Task GetHealth_ReturnsOk() {
        using var client = factory.CreateClient(new WebApplicationFactoryClientOptions {
            AllowAutoRedirect = false
        });
        using var response = await client.GetAsync("/health");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}

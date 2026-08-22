namespace KnowledgeOps.Application;

public static class DemoOrders {
    public static string? FindStatus(string orderId) => orderId switch {
        "ORD-123" => "processing",
        "ORD-456" => "shipped",
        "ORD-789" => "delivered",
        _ => null
    };
}

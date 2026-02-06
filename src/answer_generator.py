def rule_based_answer(intent):
    templates = {
        "Delivery Investigation":
            "The order was marked as delivered, but the customer reported non-receipt and a delivery investigation was initiated.",
        "Payment Issue":
            "The customer reported a payment-related issue which was investigated by support.",
        "Account Access":
            "The customer contacted support regarding account access problems."
    }
    return templates.get(intent, "Relevant information was found in the retrieved transcripts.")

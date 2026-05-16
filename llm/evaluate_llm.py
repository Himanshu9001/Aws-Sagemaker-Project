# LLM Evaluation with DeepEval
# Automatically scores Flan-T5 recommendations for quality
# Metrics: Answer Relevancy, custom actionability score
# No LLM-as-judge needed — uses rule-based + embedding metrics

import json
import boto3
import sys
import os

sys.path.insert(0, "/home/sagemaker-user/Aws-Sagemaker-Project/llm")
from prompt_manager import compile_prompt

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
FLAN_ENDPOINT = "churn-flan-t5-endpoint"
RF_ENDPOINT   = "churn-prediction-serverless"

# Test cases — input + expected output characteristics
TEST_CASES = [
    {
        "customer_id":   "high-risk-001",
        "features_csv":  "0,0,1,0,3,1,0,1,0,0,0,0,0,0,0,1,2,85.5,256.5",
        "context":       "Month-to-month contract, $85 monthly, 3 months tenure, 68% churn risk",
        "expected_type": "retention_offer",
        "should_contain_any": ["offer", "upgrade", "discount", "contract", "plan", "free", "reduce"],
        "should_not_contain": ["terminate", "cancel", "end service"],
    },
    {
        "customer_id":   "low-risk-001",
        "features_csv":  "1,0,1,1,72,1,1,0,1,1,1,1,1,1,2,0,0,45.5,3276.0",
        "context":       "Two-year contract, $45 monthly, 72 months tenure, 2% churn risk",
        "expected_type": "engagement_strategy",
        "should_contain_any": ["loyalty", "reward", "retention", "program", "customer", "engagement"],
        "should_not_contain": ["terminate", "cancel", "upgrade price"],
    },
    {
        "customer_id":   "medium-risk-001",
        "features_csv":  "0,1,0,0,12,1,0,1,0,1,0,0,1,0,0,1,2,65.5,786.0",
        "context":       "Month-to-month contract, $65 monthly, 12 months tenure, 42% churn risk",
        "expected_type": "engagement_strategy",
        "should_contain_any": ["loyalty", "reward", "retention", "program", "customer", "engagement"],
        "should_not_contain": ["terminate", "cancel"],
    },
]


def get_prediction(features_csv):
    response = runtime.invoke_endpoint(
        EndpointName=RF_ENDPOINT, ContentType="text/csv",
        Accept="application/json", Body=features_csv
    )
    result = json.loads(response["Body"].read())
    return result[0]["prediction"], result[0]["churn_probability"]


def get_recommendation(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens":50,"temperature":0.1,
                       "do_sample":False,"repetition_penalty":1.5}
    }
    response = runtime.invoke_endpoint(
        EndpointName=FLAN_ENDPOINT, ContentType="application/json",
        Accept="application/json", Body=json.dumps(payload)
    )
    result = json.loads(response["Body"].read())
    return result[0]["generated_text"]


def evaluate_relevancy(output, should_contain_any, should_not_contain):
    """Rule-based relevancy score — no LLM judge needed."""
    output_lower = output.lower()

    # Check positive keywords
    positive_score = sum(1 for kw in should_contain_any if kw in output_lower)
    positive_ratio = positive_score / len(should_contain_any)

    # Check negative keywords (hallucinations/wrong recommendations)
    negative_hits = [kw for kw in should_not_contain if kw in output_lower]
    negative_penalty = len(negative_hits) * 0.3

    # Length score — too short or too long is bad
    word_count = len(output.split())
    if word_count < 3:
        length_score = 0.2
    elif word_count <= 20:
        length_score = 1.0
    else:
        length_score = 0.7

    # Not just echoing the input
    echo_penalty = 0.5 if len(output) > 100 else 0.0

    final_score = max(0, min(1, (positive_ratio * 0.5 + length_score * 0.3) - negative_penalty - echo_penalty))
    return final_score, positive_ratio, negative_hits


def evaluate_actionability(output):
    """Is the recommendation specific and actionable?"""
    output_lower = output.lower()

    actionable_patterns = [
        "offer", "provide", "give", "upgrade", "discount",
        "free", "reduce", "add", "include", "send"
    ]
    vague_patterns = ["consider", "maybe", "perhaps", "could", "might"]

    actionable_score = sum(1 for p in actionable_patterns if p in output_lower)
    vague_score      = sum(1 for p in vague_patterns if p in output_lower)

    score = min(1.0, actionable_score * 0.3 - vague_score * 0.1)
    return max(0, score)


def run_evaluation(prompt_version="v1"):
    """Run full evaluation suite on all test cases."""
    print(f"\n{'='*65}")
    print(f"LLM Evaluation Suite — Prompt Version: {prompt_version}")
    print(f"{'='*65}")

    results = []
    total_relevancy    = 0
    total_actionability = 0

    for tc in TEST_CASES:
        prediction, probability = get_prediction(tc["features_csv"])

        # Build prompt using version manager
        if prediction == 1:
            prompt_type = "retention"
            contract    = "month-to-month"
            charges     = "$85"
            tenure      = "3 months"
            prompt, _   = compile_prompt(prompt_type, {
                "contract": contract, "charges": charges,
                "tenure": tenure, "probability": f"{probability:.0%}"
            }, version=prompt_version)
        else:
            prompt_type = "engagement"
            tenure_val  = tc["features_csv"].split(",")[4]
            prompt, _   = compile_prompt(prompt_type, {
                "tenure": f"{int(float(tenure_val))} months",
                "probability": f"{probability:.0%}"
            }, version=prompt_version)

        # Get LLM output
        raw_output = get_recommendation(prompt)
        # Flan-T5 returns full prompt + answer — extract just the answer
        output = raw_output.replace(prompt, "").strip()
        if not output:
            output = raw_output.split("?")[-1].strip() if "?" in raw_output else raw_output

        # Evaluate
        relevancy, pos_ratio, neg_hits = evaluate_relevancy(
            output, tc["should_contain_any"], tc["should_not_contain"]
        )
        actionability = evaluate_actionability(output)

        total_relevancy     += relevancy
        total_actionability += actionability

        status = "✅ PASS" if relevancy >= 0.3 else "❌ FAIL"
        results.append({
            "customer_id":    tc["customer_id"],
            "prediction":     "Churn" if prediction == 1 else "No Churn",
            "probability":    f"{probability:.0%}",
            "output":         output,
            "relevancy":      round(relevancy, 2),
            "actionability":  round(actionability, 2),
            "status":         status,
            "negative_hits":  neg_hits,
        })

        print(f"\nCustomer: {tc['customer_id']}")
        print(f"Prediction: {results[-1]['prediction']} ({results[-1]['probability']})")
        print(f"Output: {output}")
        print(f"Relevancy:     {relevancy:.2f} | Actionability: {actionability:.2f} | {status}")
        if neg_hits:
            print(f"⚠️  Negative keywords found: {neg_hits}")

    # Summary
    n = len(TEST_CASES)
    avg_relevancy     = total_relevancy / n
    avg_actionability = total_actionability / n
    pass_rate         = sum(1 for r in results if "PASS" in r["status"]) / n

    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY — Prompt v{prompt_version}")
    print(f"{'='*65}")
    print(f"Pass rate:          {pass_rate:.0%} ({sum(1 for r in results if 'PASS' in r['status'])}/{n})")
    print(f"Avg relevancy:      {avg_relevancy:.2f}/1.00")
    print(f"Avg actionability:  {avg_actionability:.2f}/1.00")
    print(f"Overall score:      {(avg_relevancy + avg_actionability) / 2:.2f}/1.00")

    return {
        "prompt_version":    prompt_version,
        "pass_rate":         pass_rate,
        "avg_relevancy":     avg_relevancy,
        "avg_actionability": avg_actionability,
        "overall_score":     (avg_relevancy + avg_actionability) / 2,
        "results":           results
    }


if __name__ == "__main__":
    # Evaluate v1 (production)
    v1_results = run_evaluation("v1")

    print(f"\n{'='*65}")
    print("Comparing v1 vs v2 prompts:")
    print(f"{'='*65}")

    # Evaluate v2 (staging)
    v2_results = run_evaluation("v2")

    # Compare
    print(f"\n{'='*65}")
    print("COMPARISON: v1 (production) vs v2 (staging)")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'v1':>8} {'v2':>8} {'Winner':>10}")
    print("-" * 55)

    metrics = ["pass_rate", "avg_relevancy", "avg_actionability", "overall_score"]
    for m in metrics:
        v1_val = v1_results[m]
        v2_val = v2_results[m]
        winner = "v2 ✅" if v2_val > v1_val else "v1" if v1_val > v2_val else "tie"
        print(f"{m:<25} {v1_val:>8.2f} {v2_val:>8.2f} {winner:>10}")

    # Recommendation
    if v2_results["overall_score"] > v1_results["overall_score"]:
        print(f"\n🚀 Recommendation: Promote v2 to production")
        print(f"   Run: promote_to_production('v2', 'retention')")
    else:
        print(f"\n✅ Keep v1 in production — v2 not yet better")

# Creates telecom retention Q&A dataset for Flan-T5 fine-tuning
# Format: instruction → response pairs
# 50 examples covering high/medium/low churn scenarios

import json
import random
import boto3

random.seed(42)

# High quality Q&A pairs for telecom retention
RETENTION_QA = [
    # High churn risk — month-to-month
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $85 monthly with 3 months tenure and 68% churn risk?",
     "Offer a 20% discount on annual contract upgrade"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $95 monthly with 2 months tenure and 75% churn risk?",
     "Provide free tech support bundle for 3 months"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $110 monthly with 1 month tenure and 82% churn risk?",
     "Offer annual contract at $79 monthly with free device protection"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $78 monthly with 4 months tenure and 65% churn risk?",
     "Discount to $60 monthly with online security addon"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $92 monthly with 3 months tenure and 71% churn risk?",
     "Upgrade to one-year contract with 15% discount"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $88 monthly with 6 months tenure and 60% churn risk?",
     "Add free tech support and reduce bill by $10 monthly"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $75 monthly with 5 months tenure and 58% churn risk?",
     "Offer annual plan at $65 monthly with streaming addon"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $102 monthly with 2 months tenure and 79% churn risk?",
     "Provide first 2 months free on annual contract"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $67 monthly with 8 months tenure and 52% churn risk?",
     "Offer loyalty discount of 10% and add online backup"),
    ("What is the best offer to prevent churn for a telecom customer on month-to-month contract paying $115 monthly with 1 month tenure and 85% churn risk?",
     "Match competitor price with annual contract guarantee"),
]

ENGAGEMENT_QA = [
    # Low churn risk — long tenure
    ("What engagement strategy should a telecom company use for a loyal customer with 72 months tenure and only 2% churn risk?",
     "Send loyalty appreciation gift and VIP support access"),
    ("What engagement strategy should a telecom company use for a loyal customer with 60 months tenure and only 5% churn risk?",
     "Enroll in platinum loyalty program with exclusive benefits"),
    ("What engagement strategy should a telecom company use for a loyal customer with 48 months tenure and only 8% churn risk?",
     "Offer early upgrade to newest device at no extra cost"),
    ("What engagement strategy should a telecom company use for a loyal customer with 36 months tenure and only 10% churn risk?",
     "Provide complimentary speed upgrade for one year"),
    ("What engagement strategy should a telecom company use for a loyal customer with 84 months tenure and only 3% churn risk?",
     "Award top-tier loyalty status with dedicated account manager"),
    ("What engagement strategy should a telecom company use for a loyal customer with 24 months tenure and only 12% churn risk?",
     "Invite to beta test new features as valued early adopter"),
    ("What engagement strategy should a telecom company use for a loyal customer with 54 months tenure and only 6% churn risk?",
     "Send personalized thank you offer with bill credit"),
    ("What engagement strategy should a telecom company use for a loyal customer with 42 months tenure and only 9% churn risk?",
     "Offer free month of premium streaming service"),
    ("What engagement strategy should a telecom company use for a loyal customer with 30 months tenure and only 11% churn risk?",
     "Provide referral bonus program invitation"),
    ("What engagement strategy should a telecom company use for a loyal customer with 66 months tenure and only 4% churn risk?",
     "Grant lifetime price lock guarantee on current plan"),
]

# Combine and augment with variations
all_pairs = RETENTION_QA + ENGAGEMENT_QA

# Create augmented dataset with price/tenure variations
augmented = []
for q, a in all_pairs:
    augmented.append({"input": q, "output": a})

    # Price variation
    for price_delta in [-10, +10]:
        new_q = q
        for price in ["$67","$75","$78","$85","$88","$92","$95","$102","$110","$115"]:
            if price in q:
                new_price = f"${int(price[1:]) + price_delta}"
                new_q = q.replace(price, new_price)
                break
        if new_q != q:
            augmented.append({"input": new_q, "output": a})

# Shuffle and take 60 examples
random.shuffle(augmented)
dataset = augmented[:60]

print(f"Dataset size: {len(dataset)} examples")
print(f"\nSample:")
print(f"Input:  {dataset[0]['input']}")
print(f"Output: {dataset[0]['output']}")

# Save locally
with open("/home/sagemaker-user/Aws-Sagemaker-Project/llm/finetuning/train_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Upload to S3
s3 = boto3.client("s3", region_name="us-east-1")
s3.put_object(
    Bucket="churn-sagemaker-artifacts",
    Key="llm/finetuning/train_data.json",
    Body=json.dumps(dataset)
)
print(f"\nDataset saved to S3: s3://churn-sagemaker-artifacts/llm/finetuning/train_data.json")

import pandas as pd

from app.services.preprocessing import normalize_dataset


def test_normalize_dataset_adapts_common_columns_and_filters_brand():
    frame = pd.DataFrame(
        {"tweet": ["  Need help https://example.com ", "Other"], "reply": ["We can help", "No"], "company": ["Acme", "Other"]}
    )
    result = normalize_dataset(frame, selected_brand="Acme")
    assert len(result) == 1
    assert result.iloc[0]["customer_message"] == "Need help"


def test_normalize_dataset_pairs_kaggle_tweets_and_filters_response_brand():
    frame = pd.DataFrame(
        {
            "tweet_id": ["1", "2", "3"],
            "author_id": ["customer", "brand-a", "brand-b"],
            "inbound": ["True", "False", "False"],
            "text": ["Need help", "We can help", "Wrong brand"],
            "response_tweet_id": ["2", "", ""],
        }
    )
    result = normalize_dataset(frame, selected_brand="brand-a")
    assert len(result) == 1
    assert result.iloc[0]["brand_response"] == "We can help"

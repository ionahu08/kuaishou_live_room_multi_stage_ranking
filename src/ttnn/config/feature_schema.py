from dataclasses import dataclass
from typing import Dict


USER_CATEGORICAL = [
    "user_id",
    "user_age_le",
    "user_gender_le",
    "user_country_le",
    "user_device_brand_le",
    "user_device_price_le",
    "fans_num_le",
    "follow_num_le",
    "accu_watch_live_cnt_le",
    "accu_watch_live_duration_le",
]

USER_NUMERIC = [
    "is_live_streamer",
    "is_photo_author",
    "user_onehot_feat0",
    "user_onehot_feat1",
    "user_onehot_feat2",
    "user_onehot_feat3",
    "user_onehot_feat4",
    "user_onehot_feat5",
    "user_onehot_feat6",
    "user_account_age",
    "user_watch_live_age",
]

ITEM_CATEGORICAL = [
    "streamer_id",
    "live_type",
    "live_content_category_le",
    "live_start_year",
    "live_start_month",
    "live_start_day",
    "live_start_hour",
    "streamer_age_le",
    "streamer_country_le",
    "streamer_device_brand_le",
    "streamer_device_price_le",
    "live_operation_tag_le",
    "fans_user_num_le",
    "fans_group_fans_num_le",
    "follow_user_num_le",
    "accu_live_cnt_le",
    "accu_live_duration_le",
    "accu_play_cnt_le",
    "accu_play_duration_le",
]

ITEM_NUMERIC = [
    "streamer_gender_le",
    "live_is_weekend",
    "title_emb_missing",
    "time_since_live_start",
    "streamer_onehot_feat0",
    "streamer_onehot_feat1",
    "streamer_onehot_feat2",
    "streamer_onehot_feat3",
    "streamer_onehot_feat4",
    "streamer_onehot_feat5",
    "streamer_onehot_feat6",
] + [f"title_emb_{i}" for i in range(128)]


@dataclass
class FeatureMeta:
    user_cat_sizes: Dict[str, int]
    item_cat_sizes: Dict[str, int]


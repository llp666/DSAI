-- 营销/流量域（6 表）：channels / ads_campaigns / ads_daily_stats / ad_conversions
--                     / promotions / traffic_events
-- 勾稽：ads_daily_stats.attributed_gmv 合计 ≈ 付费订单 GMV × attribution 系数（反推法生成）

CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE dim.channels (
    channel_id   VARCHAR PRIMARY KEY,
    channel_name VARCHAR,
    channel_type VARCHAR
);
COMMENT ON TABLE  dim.channels IS '流量渠道维度：付费渠道（信息流/搜索/CPS…）+ 自然流量渠道';
COMMENT ON COLUMN dim.channels.channel_id   IS '渠道ID，格式 channel_<类型>_<序号>，如 channel_feeds_01';
COMMENT ON COLUMN dim.channels.channel_name IS '渠道名称（中文）';
COMMENT ON COLUMN dim.channels.channel_type IS '渠道类型：feeds信息流/search搜索/cps导购/sms短信/organic自然流量';

CREATE TABLE dim.ads_campaigns (
    campaign_id   VARCHAR PRIMARY KEY,
    campaign_name VARCHAR,
    channel_id    VARCHAR,
    start_date    DATE,
    end_date      DATE,
    budget        DECIMAL(18,2)
);
COMMENT ON TABLE  dim.ads_campaigns IS '广告投放计划：一个渠道可开多个 campaign，有起止日期与预算';
COMMENT ON COLUMN dim.ads_campaigns.campaign_id   IS '投放计划ID，格式 camp_0001';
COMMENT ON COLUMN dim.ads_campaigns.campaign_name IS '投放计划名称';
COMMENT ON COLUMN dim.ads_campaigns.channel_id    IS '投放渠道，引用 dim.channels.channel_id';
COMMENT ON COLUMN dim.ads_campaigns.start_date    IS '投放开始日期';
COMMENT ON COLUMN dim.ads_campaigns.end_date      IS '投放结束日期';
COMMENT ON COLUMN dim.ads_campaigns.budget        IS '总预算（元）';

CREATE TABLE ods.ads_daily_stats (
    campaign_id    VARCHAR,
    stat_date      DATE,
    impressions    BIGINT,
    clicks         BIGINT,
    spend          DECIMAL(18,2),
    attributed_gmv DECIMAL(18,2)
);
COMMENT ON TABLE  ods.ads_daily_stats IS '广告投放日统计：曝光/点击/花费/归因GMV，一行一个campaign一天';
COMMENT ON COLUMN ods.ads_daily_stats.campaign_id    IS '投放计划ID，引用 dim.ads_campaigns.campaign_id';
COMMENT ON COLUMN ods.ads_daily_stats.stat_date      IS '统计日期';
COMMENT ON COLUMN ods.ads_daily_stats.impressions    IS '曝光次数';
COMMENT ON COLUMN ods.ads_daily_stats.clicks         IS '点击次数（CTR 行业区间 0.8%–5%）';
COMMENT ON COLUMN ods.ads_daily_stats.spend          IS '当日花费（元）= impressions × CPM/1000';
COMMENT ON COLUMN ods.ads_daily_stats.attributed_gmv IS '归因GMV（元）≈ 转化数 × 当日AOV × 0.8（勾稽校验点）';

CREATE TABLE ods.ad_conversions (
    conversion_id  VARCHAR,
    campaign_id    VARCHAR,
    order_id       VARCHAR,
    attributed_gmv DECIMAL(18,2),
    converted_at   TIMESTAMP
);
COMMENT ON TABLE  ods.ad_conversions IS '广告归因转化明细：一次归因转化一行，可回溯到订单粒度';
COMMENT ON COLUMN ods.ad_conversions.conversion_id  IS '转化记录ID，格式 ac_ + 8位零填充序号';
COMMENT ON COLUMN ods.ad_conversions.campaign_id    IS '归因的投放计划ID，引用 dim.ads_campaigns.campaign_id';
COMMENT ON COLUMN ods.ad_conversions.order_id       IS '归因订单号，引用 ods.orders.order_id';
COMMENT ON COLUMN ods.ad_conversions.attributed_gmv IS '该转化归因GMV（元），合计与 ads_daily_stats 对齐';
COMMENT ON COLUMN ods.ad_conversions.converted_at   IS '转化时间（北京时间）';

CREATE TABLE dim.promotions (
    promo_id    VARCHAR PRIMARY KEY,
    promo_name  VARCHAR,
    start_date  DATE,
    end_date    DATE,
    description VARCHAR
);
COMMENT ON TABLE  dim.promotions IS '大促活动维度：618/双11 等购物节，与 promo_days 配置对齐';
COMMENT ON COLUMN dim.promotions.promo_id    IS '活动ID，格式 promo_01';
COMMENT ON COLUMN dim.promotions.promo_name  IS '活动名称，如 618年中大促';
COMMENT ON COLUMN dim.promotions.start_date  IS '活动开始日期';
COMMENT ON COLUMN dim.promotions.end_date    IS '活动结束日期';
COMMENT ON COLUMN dim.promotions.description IS '活动说明';

CREATE TABLE ods.traffic_events (
    event_id    VARCHAR,
    user_id     VARCHAR,
    event_type  VARCHAR,
    ts          TIMESTAMP
);
COMMENT ON TABLE  ods.traffic_events IS '用户行为事件流：曝光/点击/加购/支付，漏斗四级转化系数生成（可选生成，量大）';
COMMENT ON COLUMN ods.traffic_events.event_id   IS '事件ID，格式 ev_ + 10位零填充序号';
COMMENT ON COLUMN ods.traffic_events.user_id    IS '触发用户，引用 dim.users.user_id（游客曝光行为 NULL）';
COMMENT ON COLUMN ods.traffic_events.event_type IS '事件类型：impression曝光/click点击/add_to_cart加购/pay支付（固定漏斗转化系数，全站转化率 2–3%）';
COMMENT ON COLUMN ods.traffic_events.ts         IS '事件时间（北京时间）';

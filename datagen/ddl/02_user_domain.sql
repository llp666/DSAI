-- 用户域（3 表）：dim.users / ods.user_profiles / ods.user_level_log

CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE dim.users (
    user_id       VARCHAR PRIMARY KEY,
    register_date DATE,
    city_tier     VARCHAR,
    city          VARCHAR,
    age_band      VARCHAR
);
COMMENT ON TABLE  dim.users IS '用户维度表：200k 注册用户，register_date 偏新（指数分布）';
COMMENT ON COLUMN dim.users.user_id       IS '用户ID，格式 u_ + 8位零填充序号';
COMMENT ON COLUMN dim.users.register_date IS '注册日期，偏近期（指数分布），早于首单日期';
COMMENT ON COLUMN dim.users.city_tier     IS '城市层级：一线/新一线/二线/三线/四线及以下（加权分布；可注入空值脏数据）';
COMMENT ON COLUMN dim.users.city          IS '城市名（中文）';
COMMENT ON COLUMN dim.users.age_band      IS '年龄段：18-24/25-34/35-44/45-54/55+';

CREATE TABLE ods.user_profiles (
    user_id         VARCHAR,
    first_order_date DATE,
    order_cnt       INTEGER,
    total_amt       DECIMAL(18,2)
);
COMMENT ON TABLE  ods.user_profiles IS '用户画像宽表：由订单聚合派生的用户级汇总，一行一用户';
COMMENT ON COLUMN ods.user_profiles.user_id          IS '用户ID，引用 dim.users.user_id';
COMMENT ON COLUMN ods.user_profiles.first_order_date IS '首单日期';
COMMENT ON COLUMN ods.user_profiles.order_cnt        IS '累计订单数（全历史）';
COMMENT ON COLUMN ods.user_profiles.total_amt        IS '累计消费金额（元，全历史）';

CREATE TABLE ods.user_level_log (
    log_id     VARCHAR,
    user_id    VARCHAR,
    level      INTEGER,
    changed_at TIMESTAMP
);
COMMENT ON TABLE  ods.user_level_log IS '会员等级变更日志：一次升/降级一行，可还原用户等级史';
COMMENT ON COLUMN ods.user_level_log.log_id      IS '日志ID，格式 ul_ + 8位零填充序号';
COMMENT ON COLUMN ods.user_level_log.user_id     IS '用户ID，引用 dim.users.user_id';
COMMENT ON COLUMN ods.user_level_log.level       IS '会员等级 1–5，越高消费越多';
COMMENT ON COLUMN ods.user_level_log.changed_at  IS '等级变更时间（北京时间）';

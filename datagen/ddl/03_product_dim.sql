-- 商品域与基础维度（7 表）：dim_date / categories / products / festival_calendar
--                        / product_price_log / after_sales / search_logs

CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE dim.dim_date (
    date      DATE PRIMARY KEY,
    dow       INTEGER,
    month     VARCHAR,
    is_holiday BOOLEAN,
    is_promo  BOOLEAN
);
COMMENT ON TABLE  dim.dim_date IS '日期维度：覆盖数据期逐日一行，星期/节假日/促销标记';
COMMENT ON COLUMN dim.dim_date.date       IS '日期（主键）';
COMMENT ON COLUMN dim.dim_date.dow        IS '星期几，1=周一 … 7=周日';
COMMENT ON COLUMN dim.dim_date.month      IS '所属月份 YYYY-MM，用于月度聚合';
COMMENT ON COLUMN dim.dim_date.is_holiday IS '是否法定节假日/主要节日（内置节日表）';
COMMENT ON COLUMN dim.dim_date.is_promo   IS '是否促销日或促销前后1天（对齐 promo_days 配置）';

CREATE TABLE dim.categories (
    category_id   VARCHAR PRIMARY KEY,
    category_name VARCHAR,
    category_type VARCHAR
);
COMMENT ON TABLE  dim.categories IS '商品品类维度：50 个叶子品类，按大类分配';
COMMENT ON COLUMN dim.categories.category_id   IS '品类ID，格式 CAT-01';
COMMENT ON COLUMN dim.categories.category_name IS '品类名称（中文）';
COMMENT ON COLUMN dim.categories.category_type IS '品类大类：服饰35%/电子20%/家居20%/汽配10%/其他15%';

CREATE TABLE dim.products (
    sku_id         VARCHAR PRIMARY KEY,
    sku_name       VARCHAR,
    category_id    VARCHAR,
    cost_price     DECIMAL(18,2),
    base_price     DECIMAL(18,2),
    listing_status VARCHAR,
    listing_date   DATE
);
COMMENT ON TABLE  dim.products IS '商品维度：5000 SKU，价格对数正态分布（服饰 50–500 / 电子 200–8000）';
COMMENT ON COLUMN dim.products.sku_id         IS '商品SKU，格式 SKU-0001';
COMMENT ON COLUMN dim.products.sku_name       IS '商品名称（合成拼接词，仅样例用真实名）';
COMMENT ON COLUMN dim.products.category_id    IS '所属品类，引用 dim.categories.category_id';
COMMENT ON COLUMN dim.products.cost_price     IS '成本价（元）= base_price × U(0.45, 0.65)';
COMMENT ON COLUMN dim.products.base_price     IS '基准售价（元），促销成交价可能更低';
COMMENT ON COLUMN dim.products.listing_status IS '上架状态：on_sale在售(96%)/off_sale下架/discontinued停产';
COMMENT ON COLUMN dim.products.listing_date   IS '上架日期，分布在数据期前 20 个月内';

CREATE TABLE dim.festival_calendar (
    festival_date DATE,
    festival_name VARCHAR,
    festival_type VARCHAR,
    PRIMARY KEY (festival_date, festival_name)
);
COMMENT ON TABLE  dim.festival_calendar IS '节假日明细：dim_date 衍生表，逐节日一行（RAG 叙事用）';
COMMENT ON COLUMN dim.festival_calendar.festival_date IS '节日日期';
COMMENT ON COLUMN dim.festival_calendar.festival_name IS '节日名称（中文）';
COMMENT ON COLUMN dim.festival_calendar.festival_type IS '节日类型：法定假日/购物节/传统节日';

CREATE TABLE ods.product_price_log (
    sku_id         VARCHAR,
    price          DECIMAL(18,2),
    effective_date DATE
);
COMMENT ON TABLE  ods.product_price_log IS '商品价格变更日志：一次调价一行，可还原任意日期售价';
COMMENT ON COLUMN ods.product_price_log.sku_id         IS '商品SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.product_price_log.price          IS '调价后基准售价（元）';
COMMENT ON COLUMN ods.product_price_log.effective_date IS '调价生效日期';

CREATE TABLE ods.after_sales (
    ticket_id  VARCHAR,
    order_id   VARCHAR,
    sku_id     VARCHAR,
    issue_type VARCHAR,
    status     VARCHAR,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP
);
COMMENT ON TABLE  ods.after_sales IS '售后工单：退款之外的维修/换货/投诉工单，一行一工单';
COMMENT ON COLUMN ods.after_sales.ticket_id   IS '工单号，格式 tk_ + 8位零填充序号';
COMMENT ON COLUMN ods.after_sales.order_id    IS '关联订单号，引用 ods.orders.order_id';
COMMENT ON COLUMN ods.after_sales.sku_id      IS '涉及SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.after_sales.issue_type  IS '问题类型：维修/换货/投诉/咨询';
COMMENT ON COLUMN ods.after_sales.status      IS '工单状态：open处理中/resolved已解决/closed已关闭';
COMMENT ON COLUMN ods.after_sales.created_at  IS '工单创建时间（北京时间）';
COMMENT ON COLUMN ods.after_sales.resolved_at IS '工单解决时间（北京时间），未解决为 NULL';

CREATE TABLE ods.search_logs (
    log_id       VARCHAR,
    user_id      VARCHAR,
    search_term  VARCHAR,
    result_count INTEGER,
    searched_at  TIMESTAMP
);
COMMENT ON TABLE  ods.search_logs IS '站内搜索日志：搜索词与结果数（RAG 叙事彩蛋，可选生成）';
COMMENT ON COLUMN ods.search_logs.log_id       IS '日志ID，格式 se_ + 8位零填充序号';
COMMENT ON COLUMN ods.search_logs.user_id      IS '搜索用户，引用 dim.users.user_id';
COMMENT ON COLUMN ods.search_logs.search_term  IS '搜索词（中文，含品类词与品牌词）';
COMMENT ON COLUMN ods.search_logs.result_count IS '命中结果数';
COMMENT ON COLUMN ods.search_logs.searched_at  IS '搜索时间（北京时间）';

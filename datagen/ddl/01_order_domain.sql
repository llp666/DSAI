-- 订单域（7 表）：orders / order_items / refunds / order_payments
--              / order_status_log / coupons / order_coupons
-- 规范：ods 层不建 PK/索引/FK/CHECK，唯一性与引用完整性由 validate.py 保证；
--       状态类字段允许注入脏枚举值，合法值域由语义层值字典约束。

CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE ods.orders (
    order_id     VARCHAR,
    user_id      VARCHAR,
    order_status VARCHAR,
    pay_amount   DECIMAL(18,2),
    created_at   TIMESTAMP,
    updated_at   TIMESTAMP
);
COMMENT ON TABLE  ods.orders IS '订单主表：一笔订单一行，pay_amount 为实付总额';
COMMENT ON COLUMN ods.orders.order_id     IS '订单号，格式 o_ + 8位零填充序号，如 o_00012345';
COMMENT ON COLUMN ods.orders.user_id      IS '下单用户ID，引用 dim.users.user_id';
COMMENT ON COLUMN ods.orders.order_status IS '订单状态枚举：1待支付 2已支付 3已发货 4已完成 5已退款（2025-Q1 存在历史字符串码脏数据，见值字典）';
COMMENT ON COLUMN ods.orders.pay_amount   IS '订单实付总额（元），等于 order_items 中 sale_price*qty 按单汇总';
COMMENT ON COLUMN ods.orders.created_at   IS '下单时间（北京时间，无时区）；2025-Q1 存在-8h时区错位脏数据';
COMMENT ON COLUMN ods.orders.updated_at   IS '订单最后一次状态变更时间（北京时间）；少量行存在迟到数天脏数据';

CREATE TABLE ods.order_items (
    order_id   VARCHAR,
    sku_id     VARCHAR,
    qty        INTEGER,
    sale_price DECIMAL(18,2)
);
COMMENT ON TABLE  ods.order_items IS '订单明细：一行一个订单内的一种SKU，sale_price*qty 汇总即订单实付';
COMMENT ON COLUMN ods.order_items.order_id   IS '订单号，引用 ods.orders.order_id';
COMMENT ON COLUMN ods.order_items.sku_id     IS '商品SKU，引用 dim.products.sku_id，格式 SKU-0001';
COMMENT ON COLUMN ods.order_items.qty        IS '购买数量，1–10 件';
COMMENT ON COLUMN ods.order_items.sale_price IS '成交单价（元），可能低于 base_price：促销/券';

CREATE TABLE ods.refunds (
    refund_id     VARCHAR,
    order_id      VARCHAR,
    sku_id        VARCHAR,
    refund_amount DECIMAL(18,2),
    refund_reason VARCHAR,
    refund_date   DATE
);
COMMENT ON TABLE  ods.refunds IS '退款明细：一行一条SKU级退款，关联 order_id+sku_id，可为部分退款';
COMMENT ON COLUMN ods.refunds.refund_id     IS '退款单号，格式 r_ + 8位零填充序号';
COMMENT ON COLUMN ods.refunds.order_id      IS '订单号，引用 ods.orders.order_id（仅已完成/已退款单）';
COMMENT ON COLUMN ods.refunds.sku_id        IS '退款SKU，与 order_items 的 order_id+sku_id 对应';
COMMENT ON COLUMN ods.refunds.refund_amount IS '退款金额（元）= sale_price*qty*部分退款系数(0.8–1.0)，不超过对应明细实付';
COMMENT ON COLUMN ods.refunds.refund_reason IS '退款原因：质量问题/尺码不符/七天无理由/不想要了/与描述不符/发错货，分布随品类变化';
COMMENT ON COLUMN ods.refunds.refund_date   IS '退款完成日期，晚于下单日 2+ 天（对数正态延迟）';

CREATE TABLE ods.order_payments (
    payment_id VARCHAR,
    order_id   VARCHAR,
    pay_channel VARCHAR,
    pay_amount DECIMAL(18,2),
    paid_at    TIMESTAMP
);
COMMENT ON TABLE  ods.order_payments IS '支付流水：与已支付订单 1:1，pay_amount 等于订单实付';
COMMENT ON COLUMN ods.order_payments.payment_id  IS '支付流水号，格式 p_ + 8位零填充序号';
COMMENT ON COLUMN ods.order_payments.order_id    IS '订单号，引用 ods.orders.order_id，一单一笔支付';
COMMENT ON COLUMN ods.order_payments.pay_channel IS '支付渠道：alipay/wechat_pay/union_pay/credit_card';
COMMENT ON COLUMN ods.order_payments.pay_amount  IS '支付金额（元），必须等于订单实付总额（勾稽校验点）';
COMMENT ON COLUMN ods.order_payments.paid_at     IS '支付完成时间，晚于下单时间数分钟内';

CREATE TABLE ods.order_status_log (
    log_id     VARCHAR,
    order_id   VARCHAR,
    from_status VARCHAR,
    to_status  VARCHAR,
    changed_at TIMESTAMP
);
COMMENT ON TABLE  ods.order_status_log IS '订单状态流转日志：一次状态变更一行，可还原订单生命周期';
COMMENT ON COLUMN ods.order_status_log.log_id      IS '日志ID，格式 sl_ + 8位零填充序号';
COMMENT ON COLUMN ods.order_status_log.order_id    IS '订单号，引用 ods.orders.order_id';
COMMENT ON COLUMN ods.order_status_log.from_status IS '变更前状态（首行为空）；值域同 orders.order_status';
COMMENT ON COLUMN ods.order_status_log.to_status   IS '变更后状态；值域同 orders.order_status';
COMMENT ON COLUMN ods.order_status_log.changed_at  IS '状态变更时间（北京时间）';

CREATE TABLE ods.coupons (
    coupon_id        VARCHAR,
    coupon_name      VARCHAR,
    discount_type    VARCHAR,
    threshold_amount DECIMAL(18,2),
    discount_value   DECIMAL(18,2),
    start_date       DATE,
    end_date         DATE
);
COMMENT ON TABLE  ods.coupons IS '优惠券定义：满减券/折扣券，发放与核销分开记录';
COMMENT ON COLUMN ods.coupons.coupon_id        IS '券ID，格式 CPN-0001';
COMMENT ON COLUMN ods.coupons.coupon_name      IS '券名称，如 满200减30';
COMMENT ON COLUMN ods.coupons.discount_type    IS '折扣类型：fixed满减 / percent折扣';
COMMENT ON COLUMN ods.coupons.threshold_amount IS '使用门槛金额（元），fixed 型为满 X 元';
COMMENT ON COLUMN ods.coupons.discount_value   IS '优惠值：fixed 型为减 X 元，percent 型为折扣率';
COMMENT ON COLUMN ods.coupons.start_date       IS '券生效起始日期';
COMMENT ON COLUMN ods.coupons.end_date         IS '券失效日期';

CREATE TABLE ods.order_coupons (
    oc_id           VARCHAR,
    order_id        VARCHAR,
    coupon_id       VARCHAR,
    discount_amount DECIMAL(18,2),
    used_at         TIMESTAMP
);
COMMENT ON TABLE  ods.order_coupons IS '券核销记录：一张券最多核销一单，一单可用多券';
COMMENT ON COLUMN ods.order_coupons.oc_id           IS '核销记录ID，格式 oc_ + 8位零填充序号';
COMMENT ON COLUMN ods.order_coupons.order_id        IS '订单号，引用 ods.orders.order_id';
COMMENT ON COLUMN ods.order_coupons.coupon_id       IS '券ID，引用 ods.coupons.coupon_id';
COMMENT ON COLUMN ods.order_coupons.discount_amount IS '本单实际抵扣金额（元）';
COMMENT ON COLUMN ods.order_coupons.used_at         IS '核销时间（北京时间）';

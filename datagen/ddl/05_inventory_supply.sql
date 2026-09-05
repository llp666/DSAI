-- 供应链与库存域（8 表）：suppliers / purchase_orders / purchase_order_items
--   / inventory_snapshot / warehouses / warehouse_stock / inbound_records / stock_moves
-- 勾稽：inventory_snapshot 期末库存 = 期初 + PO到货 − 销量（前向模拟保证）

CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE dim.suppliers (
    supplier_id VARCHAR PRIMARY KEY,
    supplier_name VARCHAR,
    lead_days   INTEGER
);
COMMENT ON TABLE  dim.suppliers IS '供应商维度：80 家供应商，补货提前期 7–14 天';
COMMENT ON COLUMN dim.suppliers.supplier_id   IS '供应商ID，格式 SUP-001';
COMMENT ON COLUMN dim.suppliers.supplier_name IS '供应商名称（合成）';
COMMENT ON COLUMN dim.suppliers.lead_days     IS '平均补货提前期（天）';

CREATE TABLE ods.purchase_orders (
    po_id      VARCHAR,
    supplier_id VARCHAR,
    status     VARCHAR,
    created_at TIMESTAMP,
    eta_date   DATE
);
COMMENT ON TABLE  ods.purchase_orders IS '采购单主表：库存模拟按再订货点自动生成，一行一PO';
COMMENT ON COLUMN ods.purchase_orders.po_id      IS '采购单号，格式 PO-000001';
COMMENT ON COLUMN ods.purchase_orders.supplier_id IS '供应商ID，引用 dim.suppliers.supplier_id';
COMMENT ON COLUMN ods.purchase_orders.status     IS 'PO状态：pending待确认/in_transit在途/received已入库/cancelled已取消';
COMMENT ON COLUMN ods.purchase_orders.created_at IS 'PO创建时间（北京时间）';
COMMENT ON COLUMN ods.purchase_orders.eta_date   IS '预计到货日期 = 创建日 + U(7,14) 天';

CREATE TABLE ods.purchase_order_items (
    po_id     VARCHAR,
    sku_id    VARCHAR,
    qty       INTEGER,
    unit_cost DECIMAL(18,2)
);
COMMENT ON TABLE  ods.purchase_order_items IS '采购明细：一行一个PO中的一种SKU';
COMMENT ON COLUMN ods.purchase_order_items.po_id     IS '采购单号，引用 ods.purchase_orders.po_id';
COMMENT ON COLUMN ods.purchase_order_items.sku_id    IS '采购SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.purchase_order_items.qty       IS '采购数量';
COMMENT ON COLUMN ods.purchase_order_items.unit_cost IS '采购单价（元），基于成本价浮动';

CREATE TABLE ods.inventory_snapshot (
    snapshot_date DATE,
    sku_id        VARCHAR,
    on_hand_qty   INTEGER,
    in_transit_qty INTEGER
);
COMMENT ON TABLE  ods.inventory_snapshot IS '库存周快照：一行一个SKU一周，期末库存=期初+到货−销量';
COMMENT ON COLUMN ods.inventory_snapshot.snapshot_date  IS '快照日期（周粒度，取每周最后一天）';
COMMENT ON COLUMN ods.inventory_snapshot.sku_id         IS '商品SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.inventory_snapshot.on_hand_qty    IS '期末在手库存量，可为 0（断货埋点）';
COMMENT ON COLUMN ods.inventory_snapshot.in_transit_qty IS '期末在途库存量（未到货 PO 合计）';

CREATE TABLE dim.warehouses (
    warehouse_id VARCHAR PRIMARY KEY,
    warehouse_name VARCHAR,
    region       VARCHAR
);
COMMENT ON TABLE  dim.warehouses IS '仓库维度：5 个区域仓';
COMMENT ON COLUMN dim.warehouses.warehouse_id   IS '仓库ID，格式 WH-01';
COMMENT ON COLUMN dim.warehouses.warehouse_name IS '仓库名称';
COMMENT ON COLUMN dim.warehouses.region         IS '所在大区：华北/华东/华南/西南/华中';

CREATE TABLE ods.warehouse_stock (
    warehouse_id VARCHAR,
    sku_id       VARCHAR,
    on_hand_qty  INTEGER
);
COMMENT ON TABLE  ods.warehouse_stock IS '分仓库存：SKU 在各仓的当前在手量（最新态，非快照史）';
COMMENT ON COLUMN ods.warehouse_stock.warehouse_id IS '仓库ID，引用 dim.warehouses.warehouse_id';
COMMENT ON COLUMN ods.warehouse_stock.sku_id       IS '商品SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.warehouse_stock.on_hand_qty  IS '该仓在手库存量';

CREATE TABLE ods.inbound_records (
    inbound_id   VARCHAR,
    po_id        VARCHAR,
    warehouse_id VARCHAR,
    arrived_at   TIMESTAMP,
    received_qty INTEGER
);
COMMENT ON TABLE  ods.inbound_records IS '入库记录：一次PO入库一行，与 purchase_orders 勾稽';
COMMENT ON COLUMN ods.inbound_records.inbound_id   IS '入库单号，格式 in_ + 8位零填充序号';
COMMENT ON COLUMN ods.inbound_records.po_id        IS '采购单号，引用 ods.purchase_orders.po_id';
COMMENT ON COLUMN ods.inbound_records.warehouse_id IS '入库仓库，引用 dim.warehouses.warehouse_id';
COMMENT ON COLUMN ods.inbound_records.arrived_at   IS '实际到货时间（北京时间），可能与 eta 偏差';
COMMENT ON COLUMN ods.inbound_records.received_qty IS '实际收货数量';

CREATE TABLE ods.stock_moves (
    move_id     VARCHAR,
    sku_id      VARCHAR,
    warehouse_id VARCHAR,
    move_type   VARCHAR,
    qty         INTEGER,
    moved_at    TIMESTAMP,
    ref_id      VARCHAR
);
COMMENT ON TABLE  ods.stock_moves IS '出入库流水：一笔出/入库一行，可回溯库存变化来源';
COMMENT ON COLUMN ods.stock_moves.move_id      IS '流水ID，格式 mv_ + 8位零填充序号';
COMMENT ON COLUMN ods.stock_moves.sku_id       IS '商品SKU，引用 dim.products.sku_id';
COMMENT ON COLUMN ods.stock_moves.warehouse_id IS '发生仓库，引用 dim.warehouses.warehouse_id';
COMMENT ON COLUMN ods.stock_moves.move_type    IS '流水类型：inbound入库/outbound销售出库/return退货入库';
COMMENT ON COLUMN ods.stock_moves.qty          IS '变动数量（正数，方向看 move_type）';
COMMENT ON COLUMN ods.stock_moves.moved_at     IS '变动时间（北京时间）';
COMMENT ON COLUMN ods.stock_moves.ref_id       IS '关联单据号：出库为订单号，入库为PO号';

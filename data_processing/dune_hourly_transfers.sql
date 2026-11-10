-- User-supplied acquisition query. Preserved for provenance; not executed here.
-- Ethereum only; one contract; volume scaling assumes 18 token decimals.
SELECT
    date_trunc('hour', evt_block_time) AS hour,
    COUNT(*) AS tx_count,
    SUM(value / 1e18) AS total_volume,
    COUNT(DISTINCT "from") AS active_senders,
    COUNT(DISTINCT "to") AS active_receivers
FROM erc20_ethereum.evt_Transfer
WHERE contract_address = 0xdC035D45d973E3EC169d2276DDab16f1e407384F
  AND evt_block_time >= TIMESTAMP '2018-01-01'
  AND evt_block_time < TIMESTAMP '2026-01-01'
GROUP BY 1
ORDER BY 1;

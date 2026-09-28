from crypto_pump.synergy_contracts import CONTRACT_ID, crypto_anomaly_score_payload

def test_crypto_anomaly_score_is_advisory():
    p=crypto_anomaly_score_payload(
        observation_id="o1",timestamp="2026-09-29T00:00:00Z",symbol="BTCUSDT",
        pump_probability=.73,rank=2,model_ref="pump-v1",feature_snapshot_sha256="1"*64
    )
    assert CONTRACT_ID=="synergy.trading.crypto-anomaly-score/v1"
    assert p["authorization"] is False
    assert p["requires_domain_validation"] is True

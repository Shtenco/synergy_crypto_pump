# CRYPTO PUMP V2 — multi-horizon rare-event prediction

## Новый исследовательский подход
Отличать **раннее предсказание** события от распознавания уже идущего экстремального движения. Система строит четыре будущих события роста (+10%/15min, +20%/1h, +50%/6h, +100%/24h) и совмещает CEX/DEX, кошельки, social, temporal и market regime признаки.

В [labels.py](../src/crypto_pump/v2/labels.py) future MFE строится из максимума high на полном горизонте:

$$
MFE_{t,H}=\max_{1\le j\le H}\left(\frac{High_{t+j}}{Close_t}-1\right),\quad y_t=\mathbf1\{MFE_{t,H}\ge u\}.
$$

Неполный будущий горизонт остаётся NA, а не ложной отрицательной меткой. Это конкретная защита от цензурирования последних строк.

## Модельный конвейер
[CalibratedStacker](../src/crypto_pump/v2/ensemble.py) объединяет branch scores через логистическую регрессию с balanced weights и при наличии достаточной выборки откладывает поздние 30% под isotonic calibration. [Falsification](../src/crypto_pump/v2/falsification.py) измеряет PR-AUC, ROC-AUC, precision@K, lift, концентрацию на символах, устойчивость по времени и event declustering.

$$
Lift@K=\frac{Precision@K}{BaseRate},\qquad BaseRate>0.
$$

Здесь метрики событий не равны исполнимой прибыли. Tick tape, depth, spread, листинги/делистинги, MEV, gas и рыночное воздействие являются отдельными факторами.

## Доказательные исследования
Протокол: point-in-time universe → purged train → out-of-fold branch predictions → отдельная stack/calibration sample → sealed OOS → event-level statistics. Проверять устойчивость при удалении наиболее успешных инструментов, перемешивании social features и реальном учёте издержек.

Применение — системы раннего предупреждения о рыночных аномалиях и исследовательский мониторинг манипулятивной активности. Проект не является инструментом организации пампов.

[Архитектура](../docs/V2_ARCHITECTURE.md) · [Labels](../src/crypto_pump/v2/labels.py) · [Falsification](../src/crypto_pump/v2/falsification.py)

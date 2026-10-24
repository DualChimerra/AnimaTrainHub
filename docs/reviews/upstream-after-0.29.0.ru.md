# Изменения после 0.29.0

Релиз опубликован 15 октября 2026. Проверено 24 октября 2026 по [активности](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/activity) и [сравнению с dev](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/compare/v0.29.0...de362692c3b43a809fe3d33fd7700bc21b575bc3).
Основная ветка master остаётся на 0.29.0; перечисленные 24 изменения приняты в dev.

1. [#605](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/540063149edca0e00d3d2be273133e968b139d4b) — Неподдерживаемые INT8-модели теперь явно помечены; вместо непонятных ошибок показана причина.
2. [#606](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/e676c5928a3295a3a15b529335c85581d1a2e4c4) — Добавлены FLUX.2 VAE и нужный MOSS формат кэша изображений.
3. [#607](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/bdc381a1f0e4eb880d746bb4ae4bf3c3f70d6826) — Добавлены устройство MOSS, загрузка обычных и ConvRot INT8-весов.
4. [#608](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/c47cf2bb36cc0f555d97eb66e342cfafcba532a8) — Добавлены Qwen3.5-9B и кэш текста разной длины; общая работа с кэшем выделена из Krea2.
5. [#609](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/64416f7f816573f79faac33ae7e06d9d503d8bce) — Подключено обучение MOSS с предпросмотром и LoRA. Исправлено падение ConvRot после использования общего кэша при генерации.
6. [#612](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/c325cbe031e2594e43e96ca524fae31228cfad31) — Всплывающие подсказки больше не обрезаются краями узких панелей.
7. [#610](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/db780a702c72cd4128fcc0c1204f20fc1a146a31) — Добавлены скачивание и выбор MOSS в настройках; совместимые INT8-веса больше не блокируются ошибочно.
8. [#611](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/13028579b84ac5759f7c041b3d5bd45137d1ddd2) — MOSS подключён к генерации, XY-сравнениям, регуляризационным изображениям и оценке результатов.
9. [#613](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/03a859ff16a84e6d8ab9358599b028e724602194) — Добавлены выгрузка слоёв MOSS в RAM и последовательная загрузка текстового энкодера и модели для экономии видеопамяти.
10. [#616](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/d13e400468083ae0b76801703abcd830a478e93c) — Генерация MOSS INT8, обработка текста и применение LoRA приближены к ComfyUI; исправлены детали расчёта CFG и пропуска слоёв.
11. [#614](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/9ad725afd653ee476da980221d416e0af58bb617) — Исправлен лишний возврат выгруженных слоёв Krea2 в видеопамять при работе текстового энкодера.
12. [#615](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/31fb35d49c613f05ca1d1488579b5d53b2b29f68) — Исправлен аналогичный лишний возврат слоёв Anima при декодировании через VAE.
13. [#617](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/b1adfa3f3e42afb5093ca7acf772adc29d7ad235) — Уточнены предупреждения об INT8 в XY-сравнениях и описания памяти и файлов MOSS.
14. [#618](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/0641cd2691b284a38b240feb20c9b89707cca715) — Обрезка текущего изображения больше не стирает ещё не сохранённые рамки остальных изображений.
15. [#619](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/44b84bf677d0c933ecefa09dbe1073b099e24963) — Исправлено смешивание кэшей MOSS и Anima/Krea2: проверяются формат и фактическое число каналов.
16. [#620](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/b7fe7a60206a6421936748684005523e6e657579) — Редактирование сообщений в LLM-пресете разметки больше не сбрасывает прокрутку.
17. [#621](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/643f430ea73494901fdc830a89e68a3b5df0ea1a) — Добавлены flash-attn и xformers для внимания MOSS.
18. [#622](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/0489495e2a5cc17beea9713a1932d66300952fa0) — Пояснения в настройках убраны под значок информации; у нескачанных файлов показан примерный размер.
19. [#623](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/039b3c445e06a7119d391b3bf703753d9c3008d7) — Добавлен Muon: матрицы обновляет Muon, остальные параметры — AdamW.
20. [#624](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/42f510faa6e0a74676541393ca44fb506d8c2093) — Добавлены Riemann LoRA и связанный с ним Riemannion; сохранённые файлы остаются обычными LoRA.
21. [#625](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/ebd3680649308f83fea73321e564a4a07f3c425b) — Продолжение обучения с середины эпохи больше не повторяет уже пройденные пакеты; исправлено смещение последовательности случайных чисел.
22. [#626](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/aa7d816e93f555ad7a01532562749ad7ffcb6176) — Для новых конфигов MOSS сдвиг обучающего шума изменён на 3.5, ближе к техническому отчёту. Явно сохранённые настройки не переписываются.
23. [#627](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/bc77f42555ff7d6edf97417a57c604009264236b) — Добавлено обучение MOSS с упаковкой изображений разных размеров; текст тоже учитывается в лимите размера пакета.
24. [#628](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/de362692c3b43a809fe3d33fd7700bc21b575bc3) — Ускорено MOSS-обучение при batch_size > 1: слои обрабатывают пакет вместе и не перегружаются для каждого изображения.

В активности также отражены создание 24 рабочих веток, 9 обычных и 6 принудительных обновлений веток и удаление 99 старых веток. Это обслуживание разработки, а не 138 дополнительных функций.

## Что найдено в локальном проекте

- Расчёты Muon совпадают с исходной реализацией. Он уже присутствует и в исходной схеме, и в ответе работающего сервера `/api/schema`.
- Riemannion использует ту же основную математику и пару с Riemann LoRA. Локальные дополнения защищают узкие слои и параметры, получившие градиент позже; они сохранены.
- Исправлены найденные здесь ошибки из #609 (общая матрица ConvRot после генерации) и #619 (неверный формат и каналы кэша изображений).
- Дополнительно перенесены логика INT8-генерации и применения LoRA из #616, новые значения шума из #626, упаковка разных размеров из #627 и общий проход пакета из #628. Новые конфиги используют logit_normal + shift 3.5; старые явно заданные значения сохраняются. Остальные изменения не переносились целиком.
- Запрошенные изменения формы, qfloat8 и Caption Dropout внесены отдельно. Тесты и GPU-обучение не запускались.

# Data-engineering
```mermaid
graph TD
    A[Start] --> B[Extract: Загрузка данных]
    B --> C[Transform: Предобработка данных]
    C --> D[Train: Обучение модели]
    D --> E[Evaluate: Расчет метрик]
    E --> F[Save: Сохранение результатов]
    F --> G[Upload: Выгрузка в облако]
    G --> H[End]


[Start]
  │
  ├── [Extract] Загрузка данных из источника
  │     │
  │     └── (Ошибка загрузки) → [Retry/Alert]
  │
  ├── [Transform] Предобработка данных
  │     │
  │     └── (Ошибка обработки) → [Retry/Alert]
  │
  ├── [Train] Обучение модели LogisticRegression
  │     │
  │     └── (Ошибка обучения) → [Retry/Alert]
  │
  ├── [Evaluate] Расчет метрик качества
  │     │
  │     └── (Ошибка вычислений) → [Retry/Alert]
  │
  └── [Load] Сохранение модели и метрик в облачное хранилище
        │
        └── (Ошибка сохранения) → [Retry/Alert]

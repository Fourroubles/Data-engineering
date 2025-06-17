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

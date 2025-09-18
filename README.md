# VisionDigits

Веб-приложение для распознавания рукописных и напечатанных цифр/чисел с изображений и пользовательских рисунков.

## Возможности
- Загрузка изображений с числами
- Рисование чисел мышкой прямо в браузере
- Автоматическое распознавание всех цифр на картинке
- Поддержка различных форматов изображений (JPEG, PNG, etc.)
- API для интеграции с другими приложениями

## Стек
- Backend: Python, FastAPI, PyTorch, torchvision
- Frontend: JavaScript (HTML5, Canvas API)

## Как запустить
1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Обучить модель (если файл модели отсутствует):
   ```bash
   python app/train.py
   ```

3. Запустить backend:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Открыть `frontend/index.html` в браузере

## API Документация

### Swagger UI
API можно тестировать через интерактивный Swagger-интерфейс:
- Откройте http://127.0.0.1:8000/docs в браузере
- Здесь вы можете загружать файлы изображений и тестировать все эндпоинты прямо из браузера
- Документация автоматически обновляется при изменении API

### Эндпоинты

#### POST `/predict/`
Распознавание цифры из загруженного файла изображения.

**Параметры:**
- `file` (UploadFile): Файл изображения (JPEG, PNG, etc.)

**Ответ:**
```json
{
  "prediction": "5",
  "filename": "digit.png",
  "content_type": "image/png",
  "status": "success"
}
```

#### POST `/predict/canvas/`
Распознавание цифры из изображения canvas (base64).

**Параметры:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Ответ:**
```json
{
  "prediction": "3",
  "source": "canvas",
  "status": "success"
}
```

#### GET `/health`
Проверка состояния API и модели.

**Ответ:**
```json
{
  "status": "ok",
  "message": "API и модель работают корректно"
}
```

## Поддержка Canvas

API поддерживает отправку изображений, нарисованных в HTML5 Canvas:

1. **Формат base64**: Отправляйте изображения в формате base64 через эндпоинт `/predict/canvas/`
2. **Data URL**: Поддерживается стандартный формат `data:image/png;base64,...`
3. **Автоматическое декодирование**: API автоматически распознает и обрабатывает различные форматы

### Пример использования Canvas API

```javascript
// Получение изображения из canvas
const canvas = document.getElementById('drawingCanvas');
const base64Image = canvas.toDataURL('image/png');

// Отправка на сервер
fetch('/predict/canvas/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        image: base64Image
    })
})
.then(response => response.json())
.then(data => {
    console.log('Предсказание:', data.prediction);
});
```

## Обработка ошибок

API обеспечивает детальную обработку ошибок:

- **400 Bad Request**: Неподдерживаемый формат файла или поврежденное изображение
- **503 Service Unavailable**: Модель не загружена
- **500 Internal Server Error**: Внутренние ошибки сервера

## TODO
- [ ] Минималка: загрузка и распознавание одной цифры ✅
- [ ] Рисование на canvas и отправка картинки ✅
- [ ] Выделение и распознавание нескольких чисел на изображении
- [ ] Красивая верстка frontend

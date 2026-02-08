import telebot
from telebot import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
import os

# Токен (временно для теста, потом вынесите в .env)
BOT_TOKEN = "#"
bot = telebot.TeleBot(BOT_TOKEN)

print("Загрузка модели Qwen2.5-0.5B-Instruct...")
start_time = time.time()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True  # ← КРИТИЧЕСКИ ВАЖНО!
)

print(f"✅ Модель загружена за {time.time() - start_time:.1f} сек! Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")

user_states = {}

# УЛУЧШЕННЫЙ ПРОМПТ С ПРИМЕРАМИ (few-shot learning)
def build_prompt(recipient, budget):
    examples = """
Пример 1:
Подарок для Мамы с бюджетом До 1000 ₽
Ответ: Ароматические свечи с запахом лаванды или набор травяного чая.

Пример 2:
Подарок для Другу с бюджетом 5000–15000 ₽
Ответ: Беспроводные наушники с шумоподавлением или настольная игра для компании.

Пример 3:
Подарок для Трактористу с бюджетом 2500–3000 ₽
Ответ: Термокружка с подогревом или качественные перчатки для работы.
"""
    
    prompt = f"""Ты — эксперт по подаркам. Предложи ОДНУ конкретную идею подарка.
Важно: ответ должен быть кратким (1 предложение), без нумерации, без лишних слов.

{examples}

Задача:
Подарок для {recipient} с бюджетом {budget}
Ответ:"""
    return prompt

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add("🎁 Помоги выбрать подарок")
    bot.send_message(
        message.chat.id,
        "Привет! Я помогу подобрать идеальный подарок 🎁\nНажми кнопку ниже, чтобы начать!",
        reply_markup=markup
    )

@bot.message_handler(func=lambda m: m.text == "🎁 Помоги выбрать подарок")
def choose_recipient(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for text in ["Маме", "Папе", "Бабушке", "Программисту", "Трактористу"]:
        markup.add(text)
    bot.send_message(message.chat.id, "Для кого ищем подарок?", reply_markup=markup)
    user_states[message.chat.id] = {'step': 'recipient'}

@bot.message_handler(func=lambda m: m.chat.id in user_states and user_states[m.chat.id].get('step') == 'recipient')
def choose_budget(message):
    recipient = message.text.strip()
    valid = ["Маме", "Папе", "Бабушке", "Программисту", "Трактористу"]
    if recipient not in valid:
        bot.send_message(message.chat.id, "Выберите вариант из кнопок ниже.")
        return
    
    user_states[message.chat.id] = {'step': 'budget', 'recipient': recipient}
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for text in ["До 1000 ₽", "2500–3000 ₽", "5000–15000 ₽", "50000–150000 ₽"]:
        markup.add(text)
    bot.send_message(message.chat.id, f"Выберите бюджет для подарка {recipient}:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.chat.id in user_states and user_states[m.chat.id].get('step') == 'budget')
def generate_gift(message):
    budget = message.text.strip()
    valid = ["До 1000 ₽", "2500–3000 ₽", "5000–15000 ₽", "50000–150000 ₽"]
    if budget not in valid:
        bot.send_message(message.chat.id, "Выберите бюджет из кнопок ниже.")
        return
    
    user_id = message.chat.id
    recipient = user_states[user_id]['recipient']
    
    wait_msg = bot.send_message(user_id, "✨ Ищу идеальный подарок... (5-15 сек)")
    
    try:
        start = time.time()
        
        # Генерируем УЛУЧШЕННЫЙ промпт
        prompt = build_prompt(recipient, budget)
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Генерация с защитой от обрывов
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.85,
            do_sample=True,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15  # Снижаем повторения
        )
        
        # Извлекаем только новый текст
        new_tokens = generated_ids[0, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # 🔍 ВАЛИДАЦИЯ И ОЧИСТКА ОТВЕТА
        response = response.split("\n")[0].strip()  # Берём только первую строку
        response = re.sub(r'^\d+\.\s*', '', response)  # Убираем "1. ", "2. "
        response = re.sub(r'^-\s*', '', response)      # Убираем "- "
        response = re.sub(r'[.!?]+$', '', response).strip()  # Убираем точки в конце
        
        # Если ответ слишком короткий или содержит только цифры — используем fallback
        if len(response) < 8 or re.match(r'^\d+$', response) or response in ["", "1", "2", "3"]:
            fallback_ideas = {
                ("Программисту", "5000–15000 ₽"): "Механическая клавиатура или подписка на JetBrains Toolbox",
                ("Трактористу", "5000–15000 ₽"): "Термос с подогревом или тактический фонарь",
                ("Маме", "До 1000 ₽"): "Ароматические свечи или набор травяного чая",
                ("Папе", "2500–3000 ₽"): "Мультитул или термокружка",
                ("Бабушке", "5000–15000 ₽"): "Массажёр для шеи или цифровая фоторамка"
            }
            response = fallback_ideas.get((recipient, budget), "Сертификат в магазин по интересам человека")
        
        # Формируем финальный ответ
        final_response = (
            f"🎁 Подарок для {recipient}\n"
            f"💰 Бюджет: {budget}\n\n"
            f"💡 {response}.\n\n"  # Добавляем точку для завершённости
            f"⏱ Сгенерировано за {time.time() - start:.1f} сек"
        )
        
        bot.delete_message(user_id, wait_msg.message_id)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        markup.add("🎁 Подобрать ещё подарок")
        bot.send_message(user_id, final_response, reply_markup=markup)
        
    except Exception as e:
        bot.delete_message(user_id, wait_msg.message_id)
        bot.send_message(
            user_id,
            f"⚠️ Не удалось сгенерировать. Вот проверенная идея:\n\n"
            f"💡 Подарочный сертификат в магазин по интересам — всегда уместно!",
            reply_markup=types.ReplyKeyboardMarkup(resize_keyboard=True).add("🎁 Подобрать ещё подарок")
        )
    
    if user_id in user_states:
        del user_states[user_id]

@bot.message_handler(func=lambda m: m.text == "🎁 Подобрать ещё подарок")
def restart_flow(message):
    send_welcome(message)

@bot.message_handler(func=lambda m: True)
def fallback(message):
    bot.send_message(message.chat.id, "Нажмите /start для начала подбора подарка.")

if __name__ == "__main__":
    print("\n✅ Бот запущен! Напишите /start в Telegram")
    print("💡 Первый запуск займёт 1-2 минуты (загрузка модели)")
    bot.infinity_polling()


print("Загрузка модели Qwen2.5-0.5B-Instruct...")
start_time = time.time()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # ← ДОБАВЛЕНО

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True  # ← ДОБАВЛЕНО
)

print(f"✅ Модель загружена за {time.time() - start_time:.1f} сек! Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
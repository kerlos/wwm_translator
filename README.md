<div align="center">

# 🌸 WWM Translator

### Neural Translation Tool for Where Winds Meet

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Compatible-6366F1?style=for-the-badge)](https://openrouter.ai)

[English](#english) • [Русский](#русский)

<img src="https://img.shields.io/badge/Where_Winds_Meet-Game_Localization-CD7F32?style=for-the-badge" alt="Where Winds Meet"/>

</div>

---

# English

## 📖 About

**WWM Translator** is a neural machine translation tool for localizing **"Where Winds Meet"**. It extracts texts from game files, translates them using AI models (OpenRouter, OpenAI, Anthropic, Google), and patches them back into the game.

## ✨ Features

- **Batch translation** with async processing and smart resume
- **Context-aware** — uses surrounding lines + Chinese reference for better quality
- **Special character validation** — ensures formatting stays intact
- **Multiple LLM providers** — OpenRouter, OpenAI, Anthropic, Google

## 🛠 Installation

```bash
git clone https://github.com/0niel/wwm_translator.git
cd wwm_translator

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

1. **Create `.env` file:**

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=deepseek/deepseek-chat-v3-0324
```

2. **Update `config.yaml`:**

```yaml
paths:
  game_locale_dir: "path/to/Where Winds Meet/Package/HD/oversea/locale"

languages:
  source: "en"      # Translate from English
  target: "ru"      # To Russian
  patch_lang: "de"  # Replace German locale in-game
```

## 📋 Usage

```bash
# 1. Extract texts
python main.py extract en
python main.py extract zh_cn

# 2. Translate
python main.py translate

# 3. Check progress
python main.py status

# 4. Validate & patch
python main.py validate
python main.py autopatch --install
```

### Commands

| Command | Description |
|---------|-------------|
| `extract <lang>` | Extract texts from game files |
| `translate` | Start/resume translation |
| `status` | Show progress |
| `validate` | Check special characters |
| `autopatch` | Create and install patch |
| `reset` | Reset progress |

## 🎮 About the Game

**Where Winds Meet** is an epic open-world action-adventure RPG rooted in the rich legacy of Wuxia. Set during the turbulent era of tenth-century China, you take on the role of a young sword master, uncovering forgotten truths and the mysteries of your own identity.

Explore a vibrant world filled with life—from bustling cities to hidden temples. Experience Wuxia-style traversal, master combat with classic weapons (swords, spears, fans, umbrellas), and embark on adventures alone or with up to four friends.

---

# Русский

## 📖 О проекте

**WWM Translator** — инструмент нейроперевода для локализации **«Where Winds Meet»**. Извлекает тексты из файлов игры, переводит с помощью ИИ-моделей (OpenRouter, OpenAI, Anthropic, Google) и внедряет обратно в игру.

## ✨ Возможности

- **Пакетный перевод** с асинхронной обработкой и возобновлением
- **Учёт контекста** — использует окружающие строки + китайский для качества
- **Валидация спецсимволов** — сохраняет форматирование
- **Разные LLM-провайдеры** — OpenRouter, OpenAI, Anthropic, Google

## 🛠 Установка

```bash
git clone https://github.com/0niel/wwm_translator.git
cd wwm_translator

# Установка через uv (рекомендуется)
uv sync

# Или через pip
pip install -e .
```

### Настройка

1. **Создайте `.env`:**

```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ
LLM_MODEL=deepseek/deepseek-chat-v3-0324
```

2. **Обновите `config.yaml`:**

```yaml
paths:
  game_locale_dir: "путь/к/Where Winds Meet/Package/HD/oversea/locale"

languages:
  source: "en"      # Переводим с английского
  target: "ru"      # На русский
  patch_lang: "de"  # Заменяем немецкую локаль
```

## 📋 Использование

```bash
# 1. Извлечь тексты
python main.py extract en
python main.py extract zh_cn

# 2. Перевести
python main.py translate

# 3. Проверить прогресс
python main.py status

# 4. Валидация и патч
python main.py validate
python main.py autopatch --install
```

### Команды

| Команда | Описание |
|---------|----------|
| `extract <lang>` | Извлечь тексты из файлов игры |
| `translate` | Начать/возобновить перевод |
| `status` | Показать прогресс |
| `validate` | Проверить спецсимволы |
| `autopatch` | Создать и установить патч |
| `reset` | Сбросить прогресс |

## 🎮 Об игре

**Where Winds Meet** — эпическая action-adventure RPG в открытом мире в жанре уся (Wuxia). Действие разворачивается в Китае X века. Вы — молодой мастер меча, раскрывающий забытые истины и тайны собственной личности.

Исследуйте яркий мир — от оживлённых городов до затерянных храмов. Освойте перемещение в стиле уся, сражайтесь классическим оружием (мечи, копья, веера, зонты) и отправляйтесь в приключения в одиночку или с друзьями (до 4 человек).

---

<div align="center">

## 📄 License

MIT License © 2025 [0niel](https://github.com/0niel)

Made with ❤️ for the Where Winds Meet community

</div>

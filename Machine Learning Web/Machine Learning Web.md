# Flask in Python

**Flask** is a **micro web framework** written in **Python**. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. Flask provides the essentials to build web applications, leaving the rest to the developer’s discretion.

---

## What is a Micro Framework?

The term *micro* means Flask provides only the core tools for web development:

- URL routing  
- HTTP request handling  
- Template rendering  

It does **not** include components like form validation, database abstraction layers, or authentication mechanisms out-of-the-box. These features can be added through **Flask extensions**.

---

## Core Components

Flask is built on:

- **Werkzeug**: A WSGI utility library for request and response handling.  
- **Jinja2**: A powerful templating engine that lets you render dynamic HTML content.

---

## Key Features

- **Lightweight and Flexible**  
  Minimal setup to get started, and easy to extend as needed.

- **Built-in Development Server**  
  Includes a debugger and reloader for rapid development.

- **Routing**  
  Use decorators to bind functions to specific URLs.

- **Templating**  
  Jinja2 is used to generate HTML dynamically.

- **Request Handling**  
  Easily access form data, query parameters, cookies, and JSON.

- **Extension Support**  
  Add database support, authentication, migrations, etc., through Flask extensions.

---

## Hello World Example

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)

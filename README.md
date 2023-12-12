# Minha API em REST com Machine Learning - Raquel Allemão

Projeto parte da disciplina de **Qualidade de Software, Segurança e Sistemas Inteligentes** - PUC Rio

O objetivo aqui é apresentar uma API full-stack que interaja com o modelo de Machine Learning, apresentando predições e passando no teste previsto.

---
## Como executar

1. Clonar o repositório;
2. Instalar as libs python listadas em **requirements.txt**;
3. Excecutar os comandos abaixo pelo terminal (recomendado o uso de venv).

Para instalar as libs:

``` $ pip install -r requirements.txt ```

Para executar a API:

``` $ flask run --host 0.0.0.0 --port 5000 ```

Para reiniciar o servidor após uma mudança no código (modo de desenvolvimento):

``` $ flask run --host 0.0.0.0 --port 5000 --reload ```

Para conferir o status da API, abra o http://localhost:5000/#/ no navegador!

Para executar o teste:

``` $ pytest -v back/testes.py ```

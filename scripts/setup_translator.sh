source venv/bin/activate

python3 -m pip install argostranslate
argospm update
argospm install translate-en_el
argos-translate --from en --to el "Hello World!"

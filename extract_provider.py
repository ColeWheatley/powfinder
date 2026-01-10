with open('reality_main.js', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Look for the aerial winter definition
idx = content.find('isWinter&&!s.topoMap')
if idx == -1:
    idx = content.find('isWinter && !s.topoMap')

if idx != -1:
    print("--- AERIAL WINTER DEFINITION FOUND ---")
    print(content[idx:idx+2000])
else:
    print("Not found")

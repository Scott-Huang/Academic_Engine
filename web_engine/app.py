from flask import request, Flask, render_template
from model.search import rank
#from model.logger import log

app = Flask(__name__)

@app.route('/')
def index():
    query = request.args.get('search_query', None, str)
    if not query:
        return render_template('index.html')
    
    concept_list = rank(query, k=4)
    length = len(concept_list)
    return render_template('concepts.html', query=query, concepts=concept_list, length=length)

@app.route('/<keyword>')
def keyword_page(keyword):
    return 'page for ' + keyword

if __name__ == '__main__':
    app.run(port=5000, debug=True)
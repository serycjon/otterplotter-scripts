from flask import Flask, session, render_template, request
from flask_session import Session

app = Flask(__name__)
# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.secret_key = 'seeecreeet'
app.debug = True
app.config.from_object(__name__)
Session(app)


def genetic_magic(rating_list):
    import numpy as np
    new_generation = []
    parents = []
    scores = []
    for star, score in rating_list:
        parents.append(star)
        scores.append(float(score))

    print(scores)
    scores = np.array(scores)
    scores += 2
    scores /= np.sum(scores)
    parent_ids = sorted(
        np.random.choice(len(parents),
                         size=len(parents),
                         p=scores))
    from stars import mutate_star
    for p_id in parent_ids:
        parent = parents[p_id]
        # new_generation.append(copy.deepcopy(parent))
        new_generation.append(mutate_star(parent))
    return new_generation


@app.route('/')
def reset():
    from stars import random_star, construct_star
    from svg import multilayer_svg_as_string
    stars = [random_star() for i in range(6 * 3)]
    session["configs"] = {'stars': stars}

    svgs = [multilayer_svg_as_string(construct_star(star))
            for star in stars]

    return render_template('index.html', stars=svgs)


@app.route('/optim')
def route():
    rating_list = []
    print(f"len(session['configs']): {len(session['configs'])}")
    print(f"len(session['configs']['stars']): {len(session['configs']['stars'])}")
    for i, star in enumerate(session["configs"]['stars']):
        score = request.args.get(f'd{i}', type=int)
        rating_list.append((star, score))
    stars = genetic_magic(rating_list)
    session['configs']['stars'] = stars

    from stars import construct_star
    from svg import multilayer_svg_as_string
    svgs = [multilayer_svg_as_string(construct_star(star)) for star in stars]
    return render_template('index.html', stars=svgs)


@app.route('/desc')
def desc_route():
    return session["configs"]


if __name__ == '__main__':
    app.run()

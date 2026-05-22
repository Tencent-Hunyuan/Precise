from io import BytesIO
import pickle
import traceback

from PIL import Image
from flask import Blueprint, Flask, request

from gen_eval import load_geneval


root = Blueprint("root", __name__)


def create_app():
    global inference_fn

    inference_fn = load_geneval()
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/", methods=["POST"])
def inference():
    payload = request.get_data()
    try:
        data = pickle.loads(payload)
        images = [Image.open(BytesIO(image_bytes), formats=["jpeg"]) for image_bytes in data["images"]]
        metadatas = data["meta_datas"]
        only_strict = data["only_strict"]
        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = inference_fn(
            images, metadatas, only_strict
        )
        response = {
            "scores": scores,
            "rewards": rewards,
            "strict_rewards": strict_rewards,
            "group_rewards": group_rewards,
            "group_strict_rewards": group_strict_rewards,
        }
        return pickle.dumps(response), 200
    except Exception:  # noqa: BLE001
        return traceback.format_exc().encode("utf-8"), 500

from flask import Flask, request, render_template, Response, request as flask_request
from flask_pymongo import PyMongo
import cv2
import numpy as np
import base64
import math
from datetime import datetime
import pytz
ist = pytz.timezone('Asia/Kolkata')

import os
app = Flask(__name__)
# MongoDB Atlas
mongo_uri = os.environ.get("MONGO_URI")
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable not set!")

app.config["MONGO_URI"] = mongo_uri
mongo = PyMongo(app)

latest_frame = None  # store latest frame in memory
MAX_FRAMES = 5000    # maximum frames to store in MongoDB

# Add timestamp watermark
def add_timestamp(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_frame
    img_bytes = request.data
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    frame = add_timestamp(frame)

    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    latest_frame = img_b64

    # === Rolling window logic ===
    total_frames = mongo.db.frames.count_documents({})
    if total_frames >= MAX_FRAMES:
        to_delete = total_frames - MAX_FRAMES + 1
        oldest_frames = mongo.db.frames.find().sort("timestamp", 1).limit(to_delete)
        ids_to_delete = [f['_id'] for f in oldest_frames]
        mongo.db.frames.delete_many({"_id": {"$in": ids_to_delete}})

    # Save new frame
    mongo.db.frames.insert_one({
        "timestamp": datetime.now(ist),
        "image": img_b64
    })
    return "OK", 200

def gen_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            frame_data = base64.b64decode(latest_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            continue

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pre_recordings', methods=['GET'])
def pre_recordings():
    page = int(flask_request.args.get('page', 1))
    per_page = 20

    # Filters
    filter_date = flask_request.args.get('date')  # 'YYYY-MM-DD'
    filter_hour = flask_request.args.get('hour')  # '0-23'

    query = {}
    if filter_date:
        date_start = ist.datetime.strptime(filter_date, "%Y-%m-%d")
        date_end = date_start.replace(hour=23, minute=59, second=59)
        query['timestamp'] = {"$gte": date_start, "$lte": date_end}

        if filter_hour is not None and filter_hour.isdigit():
            hour = int(filter_hour)
            query['timestamp'] = {
                "$gte": date_start.replace(hour=hour, minute=0, second=0),
                "$lte": date_start.replace(hour=hour, minute=59, second=59)
            }

    total_frames = mongo.db.frames.count_documents(query)
    total_pages = max(1, math.ceil(total_frames / per_page))

    frames_cursor = mongo.db.frames.find(query)\
        .sort("timestamp", 1)\
        .skip((page-1)*per_page)\
        .limit(per_page)

    # Organize by date
    date_dict = {}
    for f in frames_cursor:
        ts = f['timestamp']
        date_str = ts.strftime("%Y-%m-%d")
        f['timestamp'] = ts
        if date_str not in date_dict:
            date_dict[date_str] = []
        date_dict[date_str].append(f)

    return render_template(
        'pre_recordings.html',
        date_dict=date_dict,
        page=page,
        total_pages=total_pages,
        filter_date=filter_date,
        filter_hour=filter_hour
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

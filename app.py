import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from network import style_transfer, on_top_of, run_blended_style
from network2 import style_transfer as style_transfer_seg
from network3 import style_transfer as mean_style_trasnfer

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'style_image' not in request.files or 'content_image' not in request.files:
            return "Style or content image missing.", 400

        style_image = request.files['style_image']
        style_image2 = request.files['style_image2']
        content_image = request.files['content_image']

        operation = request.form.get('operation')
        start_x = request.form.get('start_x')
        start_y = request.form.get('start_y')
        width = request.form.get('width')
        height = request.form.get('height')

        # Handle empty or invalid image files
        if style_image.filename == '' or content_image.filename == '':
            return "Image files are empty.", 400

        if not allowed_file(style_image.filename) or not allowed_file(content_image.filename):
            return "Invalid file type. Only images allowed.", 400

        # Saving the images
        style_filename = secure_filename(style_image.filename)
        style_filename2 = secure_filename(style_image2.filename)
        content_filename = secure_filename(content_image.filename)


        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        style_path2 = os.path.join(app.config['UPLOAD_FOLDER'], style_filename2)
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)

        style_image.save(style_path)
        style_image2.save(style_path2)
        content_image.save(content_path)

        # Output file name and path
        output_filename = 'output.jpg' 
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Perform style transfer or partial transfer
        if operation == 'on_top_of':
            start_x = int(start_x)
            start_y = int(start_y)
            width = int(width)
            height = int(height)
            on_top_of(style_path, content_path, output_path, start_x, start_y, width, height)
        elif operation == 'blended_gram':
            style_transfer_seg(style_path, style_path2, content_path, output_path)
        elif operation == 'blended_image':
            run_blended_style(style_path, style_path2, content_path, output_path)
        elif operation == 'normal_run':
            style_transfer(style_path, content_path, output_path)
        elif operation == 'mean_gram':
            mean_style_trasnfer(style_path,style_path2,content_path,output_path)
        # Ensure the image is saved
        if not os.path.exists(output_path):
            return f"Error: Output image not saved at {output_path}"

        print(f"Image saved at: {output_path}")

        # Correct path to image for rendering
        output_image_url = url_for('static', filename=f'output/{output_filename}')
        return render_template('index.html', image_url=output_image_url)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

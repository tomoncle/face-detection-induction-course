# 识别上传对象，标注类型，并定位其准确坐标
使用java实现图像识别,并为识别对象框出准确坐标

## 效果
###### 1.选择要上传的图片
![...](https://raw.githubusercontent.com/tomoncle/img/master/face-detection-induction-course/person.jpg)

###### 2.上传操作
![...](https://raw.githubusercontent.com/tomoncle/img/master/face-detection-induction-course/1.png)

###### 3.等待上传返回
![...](https://raw.githubusercontent.com/tomoncle/img/master/face-detection-induction-course/view.png)

## 运行
```bash
$ docker run -d -p 9080:9080 tomoncleshare/face-detection-induction-course:20190512
```

## 实现
> 基于`apache-mxnet` + `springboot2`

> `springboot` 负责处理图片上传预览等功能， `apache-mxnet` 解析识别图片，并标注对象位置.

* 上传接口
```java
    /**
     * 上传图片并预览识别后的图片
     *
     * @param file 上传的文件
     * @return url for "/files/{filename:.+}/view"
     */
    @PostMapping("/detectImage")
    public ModelAndView detectImage(@RequestParam("file") MultipartFile file) throws IOException {
        // save file
        fileUpload.saveOrReplace(file);
        // upload file path
        String filePath = String.format("%s/%s", UPLOAD_ROOT_PATH, file.getOriginalFilename());
        // detect json data
        String json = ImageFileDetection.inputImage(filePath, ssdModelConfiguration.modelPrefix());
        JSONArray jsonArray = JSONArray.parseArray(json);
        // detected save path
        filePath = FileMXNetImageTools.loadImage(filePath, jsonArray.getJSONArray(0));
        final String fileName = filePath.substring(filePath.lastIndexOf("/") + 1);
        return new ModelAndView(String.format("redirect:/files/%s/view", fileName));
    }

    /**
     * 上传要识别的文件
     *
     * @return
     */
    @GetMapping("/detectImage")
    public String detectImage() {
        return " <!doctype html>\n" +
                "    <title>Upload new File</title>\n" +
                "    <h1>Upload Have Face Image File</h1>\n" +
                "    <form method=post enctype=multipart/form-data>\n" +
                "      <input type=file name=file>\n" +
                "      <input type=submit value=Upload>\n" +
                "    </form>";
    }
    
    /**
     * 图片预览
     *
     * @param filename file name
     * @return
     */
    @GetMapping(value = "/{filename:.+}/view", produces = {
            MediaType.IMAGE_PNG_VALUE,
            MediaType.IMAGE_JPEG_VALUE,
            MediaType.IMAGE_JPEG_VALUE,
            MediaType.IMAGE_GIF_VALUE})
    @ResponseBody
    public ResponseEntity<Resource> serveFileView(@PathVariable String filename) throws FileNotFoundException {
        Path load = fileUpload.load(filename);
        InputStream is = new FileInputStream(load.toFile());
        HttpHeaders headers = new HttpHeaders();
        return ResponseEntity.ok().headers(headers).body(new InputStreamResource(is));
    }
```

* 图片检测识别
```java
    private static List<Context> getContext() {
        List<Context> ctx = new ArrayList<>();
        ctx.add(Context.cpu());
        // For GPU, context.add(Context.gpu());
        return ctx;
    }

    private static String output(List<List<ObjectDetectorOutput>> output, Shape inputShape) {
        int width = inputShape.get(3);
        int height = inputShape.get(2);
        JSONArray result = new JSONArray();

        for (List<ObjectDetectorOutput> ele : output) {
            JSONArray jsonArray = new JSONArray();
            for (ObjectDetectorOutput i : ele) {
                if (i.getProbability() < IGNORE_PROBABILITY) {
                    // 如果识别准确率小于0.5则跳过
                    continue;
                }
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("class", i.getClassName());
                jsonObject.put("probability", i.getProbability());
                // 这里是按512*512 像素返回的位置值
                List<Float> locations = Arrays.asList(
                        i.getXMin() * width,
                        i.getXMax() * width,
                        i.getYMin() * height,
                        i.getYMax() * height);
                jsonObject.put("location", locations);
                jsonArray.add(jsonObject);
            }
            result.add(jsonArray);
        }

        return result.toJSONString();

    }


    public static String inputImage(String inputImagePath, String modelPathPrefix) {
        List<Context> context = getContext();
        // 1表示批量大小，在我们的例子中是单个图像。
        // 3用于图像中的通道，对于RGB图像为3
        // 512代表图像的高度和宽度
        Shape inputShape = new Shape(new int[]{1, 3, 512, 512});
        List<DataDesc> inputDescriptors = new ArrayList<>();
        // 布局指定给定的形状是基于NCHW的，NCHW是批大小、通道大小、高度和宽度
        // dtype是图像数据类型，它将是标准float32
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), Layout.NCHW()));
        BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
        ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        // topK=5表示获取识别率最高的5个对象
        final int topK = 5;
        List<List<ObjectDetectorOutput>> output = objDet.imageObjectDetect(img, topK);
        return output(output, inputShape);
    }

```

* 持久化识别后的图片到本地
```java
    /**
     * 识别对象,并标注坐标,然后将识别后的文件信息保存到本地
     * @param path 文件路径
     * @param imageDetectionJSONArray mxnet识别的标注信息
     * @return
     * @throws IOException
     */
    public static String loadImage(String path, JSONArray imageDetectionJSONArray) throws IOException {
        BufferedImage buf = buildBufferImage(path, imageDetectionJSONArray);
        final String fileType = path.substring(path.lastIndexOf(".") + 1).toLowerCase();
        final String fileOldName = path.substring(0, path.lastIndexOf(".")) + "-detect";
        final String fileNewName = String.format("%s.%s",fileOldName,fileType);
        File outputFile = new File(fileNewName);
        ImageIO.write(buf, fileType, outputFile);
        return fileNewName;
    }

    /**
     * 识别对象,并标注坐标,然后将识别后的文件信息写入BufferedImage对象
     * @param path 文件路径
     * @param imageDetectionJSONArray  mxnet识别的标注信息
     * @return
     * @throws IOException
     */
    private static BufferedImage buildBufferImage(
            String path, JSONArray imageDetectionJSONArray) throws IOException {
        // init detected box list
        List<Map<String, Integer>> boxes = new ArrayList<>();
        // init detected name list
        List<String> names = new ArrayList<>();
        // read input image file
        BufferedImage buf = ImageIO.read(new File(path));
        // get image width & height
        int width = buf.getWidth();
        int height = buf.getHeight();

        IntStream.range(0, imageDetectionJSONArray.size())
                .mapToObj(imageDetectionJSONArray::getJSONObject)
                .forEach(jsonObject -> {
                    names.add(String.format("%s: %s",
                            jsonObject.getString("class"), jsonObject.getString("probability")));
                    Map<String, Integer> map = new HashMap<>();
                    Object[] locations = jsonObject.getJSONArray("location").toArray();
                    map.put("xmin", Float.valueOf(locations[0].toString()).intValue() * width / 512);
                    map.put("xmax", Float.valueOf(locations[1].toString()).intValue() * width / 512);
                    map.put("ymin", Float.valueOf(locations[2].toString()).intValue() * height / 512);
                    map.put("ymax", Float.valueOf(locations[3].toString()).intValue() * height / 512);
                    boxes.add(map);
                });
        // add boxes and names to buf
        Image.drawBoundingBox(buf, boxes, names);

        return buf;

    }
```

##### 源码项目地址：https://github.com/tomoncle/mxnet-spring-samples 
Welcome start or fork this repo. [Follow me](https://github.com/tomoncle) get the latest developments.

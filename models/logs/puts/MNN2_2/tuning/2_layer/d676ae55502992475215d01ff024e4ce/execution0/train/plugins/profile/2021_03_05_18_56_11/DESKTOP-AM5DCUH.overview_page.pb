�	����7@����7@!����7@	�"�L\��?�"�L\��?!�"�L\��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����7@���*4@1���`�.�?A�3��7�?I�,
�(�@Y5�|�ݮg?*	�����I@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Pk�w�?!*6~E��;@)�~j�t��?1(����7@:Preprocessing2U
Iterator::Model::ParallelMapV2��0�*�?!�ޜy��7@)��0�*�?1�ޜy��7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate������?!k2fX'7@)����Mb�?1��_��/@:Preprocessing2F
Iterator::ModelU���N@�?![V��1�B@)y�&1�|?1ٛ3O��+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǺ���?!��do�<O@);�O��nr?1���i��!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��H�}m?!�2	v�@)��H�}m?1�2	v�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!`���V@)ŏ1w-!_?1`���V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{�Pk�?!Q=h䝿9@)��_�LU?13O��+�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�"�L\��?I��%\ݞX@QR�=<�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���*4@���*4@!���*4@      ��!       "	���`�.�?���`�.�?!���`�.�?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	�,
�(�@�,
�(�@!�,
�(�@B      ��!       J	5�|�ݮg?5�|�ݮg?!5�|�ݮg?R      ��!       Z	5�|�ݮg?5�|�ݮg?!5�|�ݮg?b      ��!       JGPUY�"�L\��?b q��%\ݞX@yR�=<�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ڢ��?!�ڢ��?"5
sequential/dense/MatMulMatMul0���:��?! XP�n˸?0"C
%gradient_tape/sequential/dense/MatMulMatMul?�#��?!�+A ��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul?�#��?!�K�#I��?0"7
sequential/dense_1/MatMulMatMul?�#��?!��2���?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad_ϑll��?!�œ�e�?"7
sequential/dense_2/MatMulMatMul_ϑll��?!�OW!���?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�}%R��?!���F�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?�#��?!�f��O�?"]
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNan?�#��?!�JE$pX�?Q      Y@Y>����/@aX�i��U@q��:U��W@yi�ȥee�?"�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
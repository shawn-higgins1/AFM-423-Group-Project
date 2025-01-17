�	9�Z�5@9�Z�5@!9�Z�5@	�|'�_�?�|'�_�?!�|'�_�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69�Z�5@�Ƽ�843@1�\��ky�?A�H�}�?I�H�H�@Y���S��y?*	43333�J@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�� �rh�?!��%6�?@)���S㥋?1Z���9@:Preprocessing2U
Iterator::Model::ParallelMapV2�HP��?!�#o�6@)�HP��?1�#o�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!�M����7@)46<�R�?1�Q<��84@:Preprocessing2F
Iterator::Model�ݓ��Z�?!?!��O�A@)S�!�uq{?1�L��`�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Q���?!`��;P@)��H�}m?1KFU�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!f�'�Y�@)y�&1�l?1f�'�Y�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��B�@)����Mb`?1��B�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�l����?!m�*R)A@)�~j�t�X?14鏃qC@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�|'�_�?In���R�X@QK2����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Ƽ�843@�Ƽ�843@!�Ƽ�843@      ��!       "	�\��ky�?�\��ky�?!�\��ky�?*      ��!       2	�H�}�?�H�}�?!�H�}�?:	�H�H�@�H�H�@!�H�H�@B      ��!       J	���S��y?���S��y?!���S��y?R      ��!       Z	���S��y?���S��y?!���S��y?b      ��!       JGPUY�|'�_�?b qn���R�X@yK2����?�"5
sequential/dense/MatMulMatMul!�7��X�?!!�7��X�?0"C
%gradient_tape/sequential/dense/MatMulMatMul0N/���?! C��?0"7
sequential/dense_1/MatMulMatMul%�!���?!)��È�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch0N/���?!�o�s��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad0N/���?!MYL�t�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul0N/���?! C��?"!
Adam/PowPow�����?!���x��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam]����m�?!�$s3*-�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam܅��Ǔ?!@�$��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile܅��Ǔ?!�b\
��?Q      Y@Y��/Ċ�0@a�	�N]�T@q���W@y
O �?N�?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
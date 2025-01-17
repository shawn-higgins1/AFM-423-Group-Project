�	2��A�6@2��A�6@!2��A�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2��A�6@8��_�4@1���8�j�?AHP�s�?I����@*	����̌M@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�
F%u�?!;7/}E@)��0�*�?1�3V�C@:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!h�pD��:@)vq�-�?1h�pD��:@:Preprocessing2F
Iterator::Model�z6�>�?!U*�i4C@)lxz�,C|?1���;�Y'@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!�q7��%@)"��u��q?1:�Jl@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&S��?!���N��N@)a��+ei?1�\�5<�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!څ�H(m@)�J�4a?1څ�H(m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��q���?!�B��F@)�~j�t�X?1R��3�M@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensora2U0*�S?!q�)`> @)a2U0*�S?1q�)`> @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!q�)`>�?)a2U0*�C?1q�)`>�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI9��ݮX@Q�1�ŜH�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	8��_�4@8��_�4@!8��_�4@      ��!       "	���8�j�?���8�j�?!���8�j�?*      ��!       2	HP�s�?HP�s�?!HP�s�?:	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q9��ݮX@y�1�ŜH�?�"5
sequential/dense/MatMulMatMulrK`i�&�?!rK`i�&�?0"C
%gradient_tape/sequential/dense/MatMulMatMul״z_��?!$��q�ݵ?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�Xw��?!�x�GL�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�Xw��?!�8�0]�?"7
sequential/dense_1/MatMulMatMul�Xw��?!״z_��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�V����?!�?�'�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch״z_��?!F�b����?"E
'gradient_tape/sequential/dense_1/MatMulMatMul0�K~��?!,P,�7�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamC+7�ގ?! ��%�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamC+7�ގ?!
���	�?Q      Y@Y{	�%��1@a�����T@q�.���X@yHmf���?"�
both�Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
�	9�cx07@9�cx07@!9�cx07@	er���ǔ?er���ǔ?!er���ǔ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69�cx07@��A��4@1p%;6��?A�&�����?I���%�@Y��A�Fs?*     �G@)       =2U
Iterator::Model::ParallelMapV2-C��6�?!m�w6�;;@)-C��6�?1m�w6�;;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Pk�w�?!� &W�=@)��0�*�?1�w6�;9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten���?!���Q��4@)a��+ey?1���
b*@:Preprocessing2F
Iterator::Model���&�?!,����C@)��0�*x?1�w6�;)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!��F}g�@)��H�}m?1��F}g�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!�w6�;N@)a��+ei?1���
b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!h���Q�@)�J�4a?1h���Q�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!����
8@)�~j�t�X?1&W�+�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9er���ǔ?Ir�f���X@Q��L���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��A��4@��A��4@!��A��4@      ��!       "	p%;6��?p%;6��?!p%;6��?*      ��!       2	�&�����?�&�����?!�&�����?:	���%�@���%�@!���%�@B      ��!       J	��A�Fs?��A�Fs?!��A�Fs?R      ��!       Z	��A�Fs?��A�Fs?!��A�Fs?b      ��!       JGPUYer���ǔ?b qr�f���X@y��L���?�"5
sequential/dense/MatMulMatMulԧ��$��?!ԧ��$��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul3�rZ�?!⍵�o�?0"C
%gradient_tape/sequential/dense/MatMulMatMul.����Y�?!�,����?0"7
sequential/dense_1/MatMulMatMul.����Y�?!�;7����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��1��?!O����)�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMuly��a��?!��A��?"7
sequential/dense_2/MatMulMatMuly��a��?!���@��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad7�]o�Z�?!Tc��q��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch.����Y�?!�u����?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�>47�?!=(���	�?Q      Y@Y>����/@aX�i��U@q�x��/W@y�^6z�?"�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
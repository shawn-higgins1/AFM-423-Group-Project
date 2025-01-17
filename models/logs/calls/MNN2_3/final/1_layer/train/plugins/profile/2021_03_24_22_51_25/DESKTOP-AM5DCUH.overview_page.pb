�	�r߉�5@�r߉�5@!�r߉�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�r߉�5@�D��3@12t��?A9��m4��?Iio����?*	23333�K@2U
Iterator::Model::ParallelMapV2���&�?!���ޓ�@@)���&�?1���ޓ�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ZӼ��?!�0���9@)��@��ǈ?1��U�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��0�*�?!$�L5@)�q����?1�I��(,@:Preprocessing2F
Iterator::Modelݵ�|г�?!���)G�F@)-C��6z?1��o+�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��Kv��@)����Mbp?1��Kv��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�sF���?![iָXK@)�����g?1�W]�I�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��I-S@)�J�4a?1��I-S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!AR˔��7@)Ǻ���V?1/��s7@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIj@Z�n�X@Q.��ҍ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�D��3@�D��3@!�D��3@      ��!       "	2t��?2t��?!2t��?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	io����?io����?!io����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qj@Z�n�X@y.��ҍ��?�"5
sequential/dense/MatMulMatMulChY��n�?!ChY��n�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�#�b��?!�)��/�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���{��?!5�k�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���{��?!2��%��?"7
sequential/dense_1/MatMulMatMul���{��?!�#�b���?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�#�b��?!ChY��n�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�bÛ*�?!��g0��?0"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad�����?!S���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�5w�&�?!�f�}��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�5w�&�?!	�$�p�?Q      Y@Y��/Ċ�0@a�	�N]�T@q@j���X@y�צ�2(�?"�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
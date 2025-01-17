�	�imۗ3@�imۗ3@!�imۗ3@	c�7@��?c�7@��?!c�7@��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�imۗ3@e ����1@1\�#����?A�~j�t��?I�~T��?YQ�����?*	    �H@2U
Iterator::Model::ParallelMapV2V-��?!�����=@)V-��?1�����=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!$I�$I�<@)M�St$�?1Dc}h�7@:Preprocessing2F
Iterator::Model'�����?!���>4�E@)lxz�,C|?1
^N��),@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_�L�?!N��)x95@)lxz�,C|?1
^N��),@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey�&1�l?!$I�$I�@)y�&1�l?1$I�$I�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!
^N��)L@)-C��6j?1����X@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!��S�r
@)��_vOf?1��S�r
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9c�7@��?I\+��ZX@Q{m2~���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e ����1@e ����1@!e ����1@      ��!       "	\�#����?\�#����?!\�#����?*      ��!       2	�~j�t��?�~j�t��?!�~j�t��?:	�~T��?�~T��?!�~T��?B      ��!       J	Q�����?Q�����?!Q�����?R      ��!       Z	Q�����?Q�����?!Q�����?b      ��!       JGPUYc�7@��?b q\+��ZX@y{m2~���?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulg�)mB�?!g�)mB�?"5
sequential/dense/MatMulMatMul��O��?!��~Q�6�?0"C
%gradient_tape/sequential/dense/MatMulMatMul>+��y�?!�-��y�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul>+��y�?!|ٚ�PX�?0"7
sequential/dense_1/MatMulMatMul>+��y�?!L�^�6�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�$�)X�?!���I� �?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�$�)X�?!����d��?"7
sequential/dense_2/MatMulMatMul�$�)X�?!I@��+�?0"!
Adam/PowPow>+��y�?!���.�c�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��{�R�?!����?Q      Y@Y7��Moz2@a���,daT@q�����zR@y�3�4n�?"�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�73.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
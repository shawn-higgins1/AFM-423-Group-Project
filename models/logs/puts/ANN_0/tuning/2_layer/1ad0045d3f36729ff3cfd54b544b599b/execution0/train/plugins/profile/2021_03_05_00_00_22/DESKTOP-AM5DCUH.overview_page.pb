�	ĕ�wF�3@ĕ�wF�3@!ĕ�wF�3@	���]�?���]�?!���]�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ĕ�wF�3@�\m��"2@1l��g���?A�a��4�?I�=%���?Y�f��f?*�����LI@)       =2U
Iterator::Model::ParallelMapV2���_vO�?!�+�=�?=@)���_vO�?1�+�=�?=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!�9�9@)'�����?1���Pp%5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��<,Ԋ?!	��9@)U���N@�?1;��1��2@:Preprocessing2F
Iterator::Model^K�=��?!����D@) �o_�y?1��T��(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!�+�=�?@)���_vOn?1�+�=�?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph��|?5�?!*Zs&M@)_�Q�k?1:���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!/�袋.@)HP�s�b?1/�袋.@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���]�?I�p)s�X@Q�t��
��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\m��"2@�\m��"2@!�\m��"2@      ��!       "	l��g���?l��g���?!l��g���?*      ��!       2	�a��4�?�a��4�?!�a��4�?:	�=%���?�=%���?!�=%���?B      ��!       J	�f��f?�f��f?!�f��f?R      ��!       Z	�f��f?�f��f?!�f��f?b      ��!       JGPUY���]�?b q�p)s�X@y�t��
��?�"5
sequential/dense/MatMulMatMuli3/�a��?!i3/�a��?0"C
%gradient_tape/sequential/dense/MatMulMatMull���.�?!ꓞ�"r�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMull���.�?! ��┉�?0"7
sequential/dense_1/MatMulMatMull���.�?!+DVm���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul/}�ʣ?!��5**��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�q�:�?!!���}��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���p�?!��MȖ��?"7
sequential/dense_2/MatMulMatMull���.�?!��'�9?�?0"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdam��m�ʵ�?!¦�L�z�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�h�e�?!P}9=��?0Q      Y@Y7��Moz2@a���,daT@q8����W@y��'D��?"�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
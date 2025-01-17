�	�h���5@�h���5@!�h���5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�h���5@�I)��3@1�0���?A-!�lV�?Ip|�%�@*	�����N@2U
Iterator::Model::ParallelMapV246<��?!�Of>@)46<��?1�Of>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX�5�;N�?!�t��<@)�{�Pk�?1۶m۶m5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!۶m۶m5@)46<�R�?1�JH72@:Preprocessing2F
Iterator::Modelc�ZB>�?![	�fE@)y�&1�|?1��}A'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!�����@)����Mbp?1�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��B�iޡ?!�����L@)ŏ1w-!o?1Pjo��?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�����
@)����Mb`?1�����
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�&S��?!��#�;>@)��_�LU?1Q �c�F@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(^��X@Q������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�I)��3@�I)��3@!�I)��3@      ��!       "	�0���?�0���?!�0���?*      ��!       2	-!�lV�?-!�lV�?!-!�lV�?:	p|�%�@p|�%�@!p|�%�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q(^��X@y������?�"5
sequential/dense/MatMulMatMulw�.�vb�?!w�.�vb�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�jbx�?!H�L�w&�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ْ	�ߘ?!�t��f^�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�|�}ޘ?!�W��K�?"7
sequential/dense_1/MatMulMatMul�|�}ޘ?!��w��f�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam=W�^1�?!�@Q�x��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad=W�^1�?!n�*`�?"!
Adam/PowPow�oq=�?!iYM����?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam܂/�׍?!�Q0[c��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam܂/�׍?!�I��~�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�� �hYX@y<ەP�?"�
both�Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
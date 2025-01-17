�	Z/�r��5@Z/�r��5@!Z/�r��5@	I
T�>�?I
T�>�?!I
T�>�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Z/�r��5@�쿢3@1�~�7��?A��3���?I���e��?Y���W:n?*�����YG@)       =2U
Iterator::Model::ParallelMapV2�������?!�ܿ�?�:@)�������?1�ܿ�?�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!g,5��;@)�I+��?1�֨���7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!Ho�M��6@)lxz�,C|?1�����-@:Preprocessing2F
Iterator::Model��~j�t�?!�Бr�WD@)9��v��z?1��ǜV�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!�@�*F @)ŏ1w-!o?1�@�*F @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!%/n�J�M@)�~j�t�h?1=0��(�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�@�*F@)ŏ1w-!_?1�@�*F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0�*�?!,���D9@)/n��R?1�S���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9I
T�>�?If�����X@QIvLռ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�쿢3@�쿢3@!�쿢3@      ��!       "	�~�7��?�~�7��?!�~�7��?*      ��!       2	��3���?��3���?!��3���?:	���e��?���e��?!���e��?B      ��!       J	���W:n?���W:n?!���W:n?R      ��!       Z	���W:n?���W:n?!���W:n?b      ��!       JGPUYI
T�>�?b qf�����X@yIvLռ�?�"5
sequential/dense/MatMulMatMul{͵U�?!{͵U�?0"C
%gradient_tape/sequential/dense/MatMulMatMul������?!�I�)�˵?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��b}£�?!9(�t4�?"7
sequential/dense_1/MatMulMatMul��b}£�?!yb@��N�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam������?!*{����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch������?!ړ�3so�?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad������?!��ws���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad������?!:�4�3��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul������?!���� �?0"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamF(���Ď?!7�_n�?Q      Y@Y��/Ċ�0@a�	�N]�T@qN��GsW@yO�]�Y�?"�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
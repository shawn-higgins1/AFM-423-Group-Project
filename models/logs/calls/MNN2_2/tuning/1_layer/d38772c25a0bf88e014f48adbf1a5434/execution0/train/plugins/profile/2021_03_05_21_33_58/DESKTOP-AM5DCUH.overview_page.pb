�	�]��6@�]��6@!�]��6@	"]G?²�?"]G?²�?!"]G?²�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�]��6@�xZ~��3@1���M�?A����K�?I�;2V��@YSͬ���?*	������G@2U
Iterator::Model::ParallelMapV29��v���?!8���؊;@)9��v���?18���؊;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!N�<��9@)��ZӼ�?1`[4�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate46<�R�?!A�I�7@)�<,Ԛ�}?1ي���.@:Preprocessing2F
Iterator::Model+�����?!�<�"�D@)9��v��z?18���؊+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��H�}m?!���c+�@)��H�}m?1���c+�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!A�I�WM@)y�&1�l?1<�"h8�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�/��@)ŏ1w-!_?1�/��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!N�<��9@)a2U0*�S?1�Iݗ�V@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9"]G?²�?I���T��X@Q��~t��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�xZ~��3@�xZ~��3@!�xZ~��3@      ��!       "	���M�?���M�?!���M�?*      ��!       2	����K�?����K�?!����K�?:	�;2V��@�;2V��@!�;2V��@B      ��!       J	Sͬ���?Sͬ���?!Sͬ���?R      ��!       Z	Sͬ���?Sͬ���?!Sͬ���?b      ��!       JGPUY"]G?²�?b q���T��X@y��~t��?�"5
sequential/dense/MatMulMatMul)b�n��?!)b�n��?0"C
%gradient_tape/sequential/dense/MatMulMatMul{Y���?!���B���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�x��~�?!\�gXM�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchf�9���?!�?��.��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradf�9���?!Hx�����?"7
sequential/dense_1/MatMulMatMul_�$oE�?!3{�_��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�����?!R��I�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul�=_���?!{&�K��?0"
Abs_1Abs�x��~�?!;^�3y�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�x��~�?!�����(�?Q      Y@Y��/Ċ�0@a�	�N]�T@qȵΥ"gW@y@&6s�
�?"�
both�Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
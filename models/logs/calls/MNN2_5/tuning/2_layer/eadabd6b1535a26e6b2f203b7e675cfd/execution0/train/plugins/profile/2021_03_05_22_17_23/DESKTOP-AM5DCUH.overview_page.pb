�	f�����4@f�����4@!f�����4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-f�����4@�ѯ��2@1�l�?3��?A!�rh���?Io��e�?*	     �G@2U
Iterator::Model::ParallelMapV2V-��?!l(����>@)V-��?1l(����>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!Q^Cy�9@)��_�L�?1Q^Cy�5@:Preprocessing2F
Iterator::Modelj�t��?!�k(��F@)y�&1�|?1��P^Cy-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO�?!�k(���6@)S�!�uq{?1_Cy�5,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice	�^)�p?!6��P^C!@)	�^)�p?16��P^C!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9��v���?!�5��P^K@)�����g?1����k@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!      @)ŏ1w-!_?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noII(3#�~X@Q����" @Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ѯ��2@�ѯ��2@!�ѯ��2@      ��!       "	�l�?3��?�l�?3��?!�l�?3��?*      ��!       2	!�rh���?!�rh���?!!�rh���?:	o��e�?o��e�?!o��e�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qI(3#�~X@y����" @�"5
sequential/dense/MatMulMatMul��@��h�?!��@��h�?0"C
%gradient_tape/sequential/dense/MatMulMatMulAa���S�?!�}�Va^�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulm9`�Τ?!?�V����?0"7
sequential/dense_1/MatMulMatMulm9`�Τ?!��d[��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulk?2�5�?!]�q���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulP�G8��?!g~z�>��?"7
sequential/dense_2/MatMulMatMulP�G8��?!��A�I��?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamm9`�Δ?!�REi8��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradm9`�Δ?!Z�H',�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�#Va6�?!w��'�%�?Q      Y@Y7��Moz2@a���,daT@qI��3-xX@y2�,�A�?"�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
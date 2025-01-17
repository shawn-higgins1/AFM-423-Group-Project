�	�)�n��3@�)�n��3@!�)�n��3@	��
=�?��
=�?!��
=�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�)�n��3@�n�1D2@1P�i4��?A���S㥫?I��>��?Yt�Lh�Xr?*	33333sJ@2U
Iterator::Model::ParallelMapV2���QI�?!��c:;@)���QI�?1��c:;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX9��v��?!�
J��L=@)��0�*�?1u.�eN6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!�r��7@)'�����?1���c:4@:Preprocessing2F
Iterator::Model�g��s��?!H��	D@)lxz�,C|?1!Y�B*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!$Vn\1�@)���_vOn?1$Vn\1�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!��DR�M@)��H�}m?1#8̺�8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!%t�û@)ŏ1w-!_?1%t�û@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��
=�?I�*Ǔ`�X@Q}%
�K�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�n�1D2@�n�1D2@!�n�1D2@      ��!       "	P�i4��?P�i4��?!P�i4��?*      ��!       2	���S㥫?���S㥫?!���S㥫?:	��>��?��>��?!��>��?B      ��!       J	t�Lh�Xr?t�Lh�Xr?!t�Lh�Xr?R      ��!       Z	t�Lh�Xr?t�Lh�Xr?!t�Lh�Xr?b      ��!       JGPUY��
=�?b q�*Ǔ`�X@y}%
�K�?�"5
sequential/dense/MatMulMatMul��ݺ$m�?!��ݺ$m�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���=�?!9�_�0.�?0"7
sequential/dense_1/MatMulMatMul���=�?!��Pu�%�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul5���q�?!D�[�d��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulv�tUq�?!�8�S�K�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrada�]�?!�{P�i�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMula�]�?!`�k}��?"7
sequential/dense_2/MatMulMatMula�]�?!��c{���?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam���=�?!N��L��?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�$�#���?!��O�J�?0Q      Y@Y7��Moz2@a���,daT@q�nK�#�W@y��Pu�%�?"�
both�Your program is POTENTIALLY input-bound because 91.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
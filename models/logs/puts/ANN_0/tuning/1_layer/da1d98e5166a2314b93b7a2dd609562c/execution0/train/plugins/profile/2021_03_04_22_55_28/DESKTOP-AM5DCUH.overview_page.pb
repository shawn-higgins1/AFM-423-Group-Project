�	����{�1@����{�1@!����{�1@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����{�1@FzQ�_U0@1��r��{�?AB"m�OT�?I�����/�?*	�����G@2U
Iterator::Model::ParallelMapV2_�Qڋ?!���D�o=@)_�Qڋ?1���D�o=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�HP��?!��9T,h:@)��_�L�?1�Cł6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<�R�?!O_ƫӗ7@)y�&1�|?1N6�d�M.@:Preprocessing2F
Iterator::ModelM�O��?!&rk��E@)F%u�{?1$I�$I�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!P���� @)�q����o?1P���� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9��v���?!ۍ��v#L@)�����g?1�n��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!���8+@)��H�}]?1���8+@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�v��?�X@Q���&=`�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	FzQ�_U0@FzQ�_U0@!FzQ�_U0@      ��!       "	��r��{�?��r��{�?!��r��{�?*      ��!       2	B"m�OT�?B"m�OT�?!B"m�OT�?:	�����/�?�����/�?!�����/�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�v��?�X@y���&=`�?�"5
sequential/dense/MatMulMatMulƾ�݄y�?!ƾ�݄y�?0"C
%gradient_tape/sequential/dense/MatMulMatMuls6�7��?!�h�Sޔ�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�üEܞ?!0������?"7
sequential/dense_1/MatMulMatMul�üEܞ?!%����?0"!
Adam/PowPows6�7��?!`�k����?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrads6�7��?!�������?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam֍��)��?!i�����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam֍��)��?!a΀L'�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam֍��)��?!��H�O�?"#
Adam/addAddV2֍��)��?!�rû�w�?Q      Y@Y�M�_{4@a��(�S@qL[~�fsX@y�]�`���?"�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
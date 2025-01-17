�	/l�V^�6@/l�V^�6@!/l�V^�6@	��_Af�?��_Af�?!��_Af�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6/l�V^�6@Tr3�P4@1iT�d��?A�΢w*�?I��im��?Y :̗`o?*	433333M@2U
Iterator::Model::ParallelMapV2�N@aÓ?!bĈ#�@@)�N@aÓ?1bĈ#�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?! ?~���6@)M�St$�?1�-[�lY3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���_vO�?!{��իW9@)�g��s��?1iҤI�&2@:Preprocessing2F
Iterator::Model}гY���?!�#F�XF@)_�Q�{?1�4iҤI'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!F�1b�@)�J�4q?1F�1b�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!x�ܹs�K@)a��+ei?1�s�Ν;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!F�1b�@)�J�4a?1F�1b�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvq�-�?!���;@)����MbP?1[�lٲe�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��_Af�?I�����X@Q��}���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Tr3�P4@Tr3�P4@!Tr3�P4@      ��!       "	iT�d��?iT�d��?!iT�d��?*      ��!       2	�΢w*�?�΢w*�?!�΢w*�?:	��im��?��im��?!��im��?B      ��!       J	 :̗`o? :̗`o?! :̗`o?R      ��!       Z	 :̗`o? :̗`o?! :̗`o?b      ��!       JGPUY��_Af�?b q�����X@y��}���?�"5
sequential/dense/MatMulMatMul����{�?!����{�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�&e�ߤ?!\dy��-�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchO�p���?!�kų�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulO�p���?!��Xܜ�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�&e�ߔ?!d����8�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�&e�ߔ?!��f���?"7
sequential/dense_1/MatMulMatMul�&e�ߔ?!�zG�p�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�o��O�?!�!��e�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�o��O�?!��>�Z�?"!
Adam/PowPow�o��O�?!�o��O�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�'s�W@yqo��N�?"�
both�Your program is POTENTIALLY input-bound because 90.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 
�	b�[>�"6@b�[>�"6@!b�[>�"6@	t��TO�?t��TO�?!t��TO�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6b�[>�"6@�󬤽3@1<�R�!��?AM�St$�?I>^H��p @YM��(#.�?*	effff�H@2U
Iterator::Model::ParallelMapV2%u��?!^2;D�=@)%u��?1^2;D�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!�!ۂ�*:@)�g��s��?1��nwB�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I+��?!�,��O6@)���Q�~?1��u�m.@:Preprocessing2F
Iterator::Model��_vO�?!�c@�E@)lxz�,C|?1*�D��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�\��e@)y�&1�l?1�\��e@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!M����L@)Ǻ���f?1���Mҷ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!
�-H�@)HP�s�b?1
�-H�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!r+�<W8@)����MbP?1Lfǀ(: @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9t��TO�?I �P���X@Q�,d���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�󬤽3@�󬤽3@!�󬤽3@      ��!       "	<�R�!��?<�R�!��?!<�R�!��?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	>^H��p @>^H��p @!>^H��p @B      ��!       J	M��(#.�?M��(#.�?!M��(#.�?R      ��!       Z	M��(#.�?M��(#.�?!M��(#.�?b      ��!       JGPUYt��TO�?b q �P���X@y�,d���?�"5
sequential/dense/MatMulMatMul9Y��b��?!9Y��b��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�k���?!}b��@R�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�F���B�?!)4]���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�X�o�?!+eF&��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�k���?!�R '��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�k���?!@��jR�?"7
sequential/dense_1/MatMulMatMul�k���?!�-Կ���?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�!7f.��?!��7����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�!7f.��?!�����?"!
Adam/PowPow�!7f.��?!��rG��?Q      Y@Y��/Ċ�0@a�	�N]�T@q��`�qV@y/�*�i�?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�88.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 